import hashlib
import json
import re
import shutil
import tempfile
import time
import unicodedata
from bisect import bisect_right
from dataclasses import asdict
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import httpx
from pypdf import PdfReader
from qdrant_client import QdrantClient, models

from .config import AppError, Settings
from .embeddings import Embeddings
from .models import Chunk, SourceDocument

DOCUMENTS = (
    SourceDocument(
        "rmf",
        "NIST AI RMF 1.0",
        "nist-ai-rmf-1.0.pdf",
        "https://nvlpubs.nist.gov/nistpubs/ai/nist.ai.100-1.pdf",
        "https://www.nist.gov/itl/ai-risk-management-framework",
        first_body_page=6,
    ),
    SourceDocument(
        "genai",
        "NIST AI RMF: Generative AI Profile",
        "nist-genai-profile.pdf",
        "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf",
        "https://www.nist.gov/publications/"
        "artificial-intelligence-risk-management-framework-generative-artificial-intelligence",
        first_body_page=5,
        last_body_page=57,
    ),
)
COLLECTION = "nist_rmf"
INDEX_VERSION = 2
HEADING = re.compile(
    r"^(?:\d{1,2}(?:\.\d{1,2})*\.?|[A-D](?:\.\d{1,2})+\.?|Appendix [A-D][.:]?)"
    r"\s+[A-Z][A-Za-z &(),/\-–—]{2,85}$"
)


def download_pdf(document: SourceDocument, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / document.filename
    if path.exists():
        validate_pdf(path)
        return path
    temporary = path.with_suffix(".part")
    try:
        with httpx.Client(timeout=45, follow_redirects=True) as client:
            for attempt in range(3):
                try:
                    with client.stream("GET", document.url) as response:
                        response.raise_for_status()
                        size = 0
                        with temporary.open("wb") as output:
                            for block in response.iter_bytes():
                                size += len(block)
                                if size > 40 * 1024 * 1024:
                                    raise AppError("The PDF download exceeded 40 MB.")
                                output.write(block)
                    validate_pdf(temporary)
                    temporary.replace(path)
                    return path
                except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                    retryable = not isinstance(
                        exc, httpx.HTTPStatusError
                    ) or exc.response.status_code in {429, 500, 502, 503, 504}
                    if not retryable or attempt == 2:
                        raise
                    time.sleep(2**attempt)
    except (httpx.HTTPError, OSError) as exc:
        raise AppError(
            f"Could not download {document.title}. Download it from {document.url} "
            f"or {document.publication_url}, save it as {path}, then run ingest again."
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    raise AppError(f"Download failed: {document.title}")


def validate_pdf(path: Path):
    try:
        with path.open("rb") as file:
            if file.read(5) != b"%PDF-":
                raise ValueError("not a PDF")
        if not PdfReader(path).pages:
            raise ValueError("empty PDF")
    except Exception as exc:
        raise AppError(f"{path} is not a readable PDF. Delete it and download it again.") from exc


def clean_page(raw: str) -> tuple[str, str | None]:
    printed_page = None
    raw = unicodedata.normalize("NFKC", raw)
    lines = raw.replace("\u00ad", "").replace("\x02", "").splitlines()
    kept = []
    for i, line in enumerate(lines):
        line = re.sub(r"[ \t]+", " ", line).strip()
        footer = re.fullmatch(r"Page\s+([ivxlcdm]+|\d+)", line, re.I)
        if footer:
            printed_page = footer.group(1)
            continue
        if (i <= 3 or i >= len(lines) - 3) and re.fullmatch(r"\d{1,3}", line):
            printed_page = line
            continue
        if re.match(r"^NIST (?:AI|Trustworthy and Responsible AI)\s+\d", line):
            continue
        kept.append(line)
    text = "\n".join(kept)
    text = re.sub(r"(\w)-\n(?=[a-z])", r"\1", text)
    return text.strip(), printed_page


def split_tokens(text: str, tokenizer, size=220, overlap=32):
    """Use exact WordPiece offsets, preferring a nearby sentence or line ending."""
    if not 0 <= overlap < size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")
    offsets = tokenizer.encode(text, add_special_tokens=False).offsets
    ends = [end for _, end in offsets]
    start = 0
    while start < len(offsets):
        stop = min(start + size, len(offsets))
        if stop < len(offsets):
            left, right = offsets[start][0], offsets[stop - 1][1]
            boundaries = list(re.finditer(r"\n|[.!?;]\s", text[left:right]))
            if boundaries and boundaries[-1].end() > (right - left) * 0.65:
                stop = max(start + 1, bisect_right(ends, left + boundaries[-1].end()))
        yield text[offsets[start][0] : offsets[stop - 1][1]].strip()
        if stop == len(offsets):
            break
        start = max(start + 1, stop - overlap)


def extract_chunks(document: SourceDocument, path: Path, tokenizer, settings: Settings):
    reader = PdfReader(path)
    section = None
    chunks = []
    for page_number, page in enumerate(reader.pages, start=1):
        if page_number < document.first_body_page:
            continue
        if document.last_body_page and page_number > document.last_body_page:
            continue
        text, printed_page = clean_page(page.extract_text() or "")
        if len(text) < 50:
            continue
        # Contents and figure lists can look relevant but omit the actual explanation.
        if re.search(r"^(?:Table of Contents|List of (?:Tables|Figures))\s*$", text, re.M | re.I):
            continue
        blocks = []
        current = []
        for line in text.splitlines():
            if HEADING.fullmatch(line.strip()):
                if current:
                    blocks.append((section, "\n".join(current)))
                section = line.strip()
                current = []
            current.append(line)
        if current:
            blocks.append((section, "\n".join(current)))
        for block_section, block in blocks:
            for chunk_text in split_tokens(
                block, tokenizer, settings.chunk_tokens, settings.chunk_overlap
            ):
                if not chunk_text:
                    continue
                identity = f"{document.id}:{page_number}:{block_section}:{chunk_text}"
                chunks.append(
                    Chunk(
                        id=str(uuid5(NAMESPACE_URL, identity)),
                        text=chunk_text,
                        document_id=document.id,
                        document=document.title,
                        pdf_page=page_number,
                        printed_page=printed_page,
                        section=block_section,
                        url=document.url,
                    )
                )
    if not chunks:
        raise AppError(f"No searchable text found in {path}. Scanned PDFs need OCR.")
    return chunks, len(reader.pages)


def build_index(settings: Settings, *, force=False, embedder=None):
    paths = [download_pdf(doc, settings.data_dir / "raw") for doc in DOCUMENTS]
    fingerprints = {
        doc.id: hashlib.sha256(path.read_bytes()).hexdigest() for doc, path in zip(DOCUMENTS, paths)
    }
    signature = {
        "version": INDEX_VERSION,
        "model": settings.embedding_model,
        "chunk_tokens": settings.chunk_tokens,
        "overlap": settings.chunk_overlap,
        "documents": fingerprints,
    }
    manifest_path = settings.index_dir / "manifest.json"
    if manifest_path.exists() and not force:
        try:
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            if old["signature"] == signature and (settings.index_dir / "chunks.json").exists():
                return {**old, "reused": True}
        except (ValueError, KeyError):
            pass

    embedder = embedder or Embeddings(settings)
    chunks, page_counts = [], {}
    for doc, path in zip(DOCUMENTS, paths):
        document_chunks, page_count = extract_chunks(doc, path, embedder.tokenizer, settings)
        chunks.extend(document_chunks)
        page_counts[doc.id] = page_count
    vectors = embedder.documents([chunk.text for chunk in chunks])
    if len(vectors) != len(chunks) or not vectors:
        raise AppError("The embedding model returned an unexpected number of vectors.")
    manifest = {
        "signature": signature,
        "chunks": len(chunks),
        "dimensions": len(vectors[0]),
        "pages": page_counts,
    }
    stage = Path(tempfile.mkdtemp(prefix=".index-", dir=settings.data_dir))
    client = None
    try:
        client = QdrantClient(path=str(stage / "qdrant"))
        client.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(
                size=len(vectors[0]),
                distance=models.Distance.COSINE,
            ),
        )
        for start in range(0, len(chunks), 64):
            client.upsert(
                COLLECTION,
                points=[
                    models.PointStruct(id=c.id, vector=v, payload=asdict(c))
                    for c, v in zip(chunks[start : start + 64], vectors[start : start + 64])
                ],
            )
        client.close()
        client = None
        (stage / "chunks.json").write_text(
            json.dumps([asdict(c) for c in chunks], ensure_ascii=False), encoding="utf-8"
        )
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        previous = settings.data_dir / "index.previous"
        if previous.exists():
            shutil.rmtree(previous)
        if settings.index_dir.exists():
            settings.index_dir.rename(previous)
        try:
            stage.rename(settings.index_dir)
        except OSError:
            if previous.exists():
                previous.rename(settings.index_dir)
            raise
        if previous.exists():
            shutil.rmtree(previous)
    finally:
        if client is not None:
            client.close()
        if stage.exists():
            shutil.rmtree(stage)
    return {**manifest, "reused": False}
