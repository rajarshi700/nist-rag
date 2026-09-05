import json
import re
from collections import defaultdict

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from .config import AppError, Settings
from .embeddings import Embeddings
from .ingest import COLLECTION, INDEX_VERSION
from .models import Chunk, Hit

STOP_WORDS = set(
    (
        "a an and are as at be by for from how in is it of on or that the to was what which with"
    ).split()
)
DOMAIN_WORDS = {"nist", "ai", "rmf", "framework", "profile", "generative"}


def terms(text: str) -> list[str]:
    return [word for word in re.findall(r"[a-z0-9]+", text.lower()) if word not in STOP_WORDS]


def fuse_ranks(rankings: list[list[str]], k=60) -> dict[str, float]:
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, chunk_id in enumerate(dict.fromkeys(ranking), start=1):
            scores[chunk_id] += 1 / (k + rank)
    return dict(scores)


class Retriever:
    def __init__(self, settings: Settings, embedder=None):
        self.settings = settings
        self.client = None
        try:
            manifest = json.loads((settings.index_dir / "manifest.json").read_text("utf-8"))
            if manifest["signature"]["model"] != settings.embedding_model:
                raise AppError(
                    "The embedding model changed. Run ingest --force to rebuild the index."
                )
            if manifest["signature"]["version"] != INDEX_VERSION:
                raise AppError("The index format changed. Run ingest --force.")
            self.chunks = [
                Chunk(**item)
                for item in json.loads((settings.index_dir / "chunks.json").read_text("utf-8"))
            ]
            if not self.chunks:
                raise ValueError("empty index")
            self.client = QdrantClient(path=str(settings.index_dir / "qdrant"))
            count = self.client.count(COLLECTION, exact=True).count
            if count != len(self.chunks):
                raise ValueError("index and metadata differ")
        except AppError:
            raise
        except Exception as exc:
            if self.client:
                self.client.close()
            raise AppError(
                "The index is missing, damaged, or open in another process. "
                "Close other instances and run: python -m nist_rag ingest --force"
            ) from exc
        try:
            self.embedder = embedder or Embeddings(settings)
            self.by_id = {chunk.id: chunk for chunk in self.chunks}
            self.bm25 = BM25Okapi(
                [terms(f"{chunk.section or ''} {chunk.text}") for chunk in self.chunks]
            )
        except Exception:
            self.close()
            raise

    def search(self, query: str) -> list[Hit]:
        if not query.strip():
            return []
        pool_size = min(len(self.chunks), max(20, self.settings.top_k * 3))
        points = self.client.query_points(
            collection_name=COLLECTION,
            query=self.embedder.query(query),
            limit=pool_size,
            with_payload=False,
        ).points
        dense = {str(point.id): float(point.score) for point in points}
        # These words occur throughout this particular corpus and otherwise promote titles.
        query_terms = terms(query)
        focused_terms = [word for word in query_terms if word not in DOMAIN_WORDS]
        scores = self.bm25.get_scores(focused_terms or query_terms)
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:pool_size]
        lexical = {self.chunks[i].id: float(scores[i]) for i in indices if scores[i] > 0}
        fused = fuse_ranks([list(dense), list(lexical)])
        ordered = sorted(fused, key=fused.get, reverse=True)

        # Overlap should help recall without using every context slot for the same passage.
        selected = []
        for chunk_id in ordered:
            chunk = self.by_id[chunk_id]
            tokens = set(terms(chunk.text))
            near_duplicate = False
            for hit in selected:
                if (chunk.document_id, chunk.pdf_page) != (
                    hit.chunk.document_id,
                    hit.chunk.pdf_page,
                ):
                    continue
                other = set(terms(hit.chunk.text))
                if len(tokens & other) / max(1, len(tokens | other)) > 0.85:
                    near_duplicate = True
                    break
            if not near_duplicate:
                selected.append(
                    Hit(chunk, fused[chunk_id], dense.get(chunk_id), lexical.get(chunk_id))
                )
            if len(selected) == self.settings.top_k:
                break
        return selected

    def close(self):
        if self.client:
            self.client.close()
            self.client = None
