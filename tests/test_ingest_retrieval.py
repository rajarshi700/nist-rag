from dataclasses import replace
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import httpx
import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from nist_rag.config import AppError, Settings
from nist_rag.ingest import (
    DOCUMENTS,
    build_index,
    clean_page,
    download_pdf,
    extract_chunks,
    split_tokens,
)
from nist_rag.retrieval import Retriever, fuse_ranks


@pytest.fixture(autouse=True)
def fixture_documents(monkeypatch):
    # Synthetic PDFs contain body text from their first page.
    monkeypatch.setattr(
        "nist_rag.ingest.DOCUMENTS",
        tuple(replace(doc, first_body_page=1, last_body_page=None) for doc in DOCUMENTS),
    )


class TinyEmbeddings:
    """Cheap vectors for storage integration tests, not semantic-quality evaluation."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def documents(self, texts):
        return [self.query(text) for text in texts]

    def query(self, text):
        text = text.lower()
        return [float(text.count("govern")), float(text.count("confabulation")), 0.1]


def make_pdf(path: Path, pages: list[list[str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter()
    for lines in pages:
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
        commands = ["BT /F1 12 Tf 18 TL 45 770 Td"]
        for line in lines:
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            commands.append(f"({escaped}) Tj T*")
        stream = DecodedStreamObject()
        stream.set_data(("\n".join(commands) + "\nET").encode("ascii"))
        page[NameObject("/Contents")] = stream
    writer.write(path)


def test_chunks_stay_within_pages_with_stable_ids_and_sections(tmp_path, tokenizer):
    path = tmp_path / "source.pdf"
    make_pdf(
        path,
        [
            [
                "5. AI RMF Core",
                "The four functions are GOVERN, MAP, MEASURE, and MANAGE.",
                "Page 20",
            ],
            ["5.1 Govern", "GOVERN is cross-cutting and informs the other functions.", "Page 21"],
        ],
    )
    source = replace(DOCUMENTS[0], first_body_page=1)
    chunks, count = extract_chunks(source, path, tokenizer, Settings())
    assert count == 2
    assert {chunk.pdf_page for chunk in chunks} == {1, 2}
    assert chunks[-1].section == "5.1 Govern"
    assert chunks[-1].printed_page == "21"
    again, _ = extract_chunks(source, path, tokenizer, Settings())
    assert [c.id for c in chunks] == [c.id for c in again]
    assert all("Page 2" not in c.text for c in chunks)


def test_token_boundaries_cover_long_text_with_overlap(tokenizer):
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = list(split_tokens(text, tokenizer, size=220, overlap=32))
    assert len(chunks) > 1
    assert all(len(tokenizer.encode(c, add_special_tokens=False).ids) <= 220 for c in chunks)
    assert all(word in " ".join(chunks).split() for word in text.split())
    assert set(chunks[0].split()) & set(chunks[1].split())


def test_normalizes_pdf_ligatures_and_printed_header():
    text, page = clean_page(" \n5\n2.2 Confabulation\nThe proﬁle discusses conﬁdence.")
    assert page == "5"
    assert "profile" in text and "confidence" in text


def test_ingestion_is_idempotent_and_qdrant_survives_reopening(tmp_path, tokenizer):
    settings = Settings(data_dir=tmp_path, top_k=2)
    for doc, body in zip(
        DOCUMENTS,
        [
            "GOVERN is cross-cutting. The four functions are GOVERN, MAP, MEASURE, MANAGE.",
            "Confabulation means confidently stated but erroneous or false content "
            "from generative AI.",
        ],
    ):
        make_pdf(tmp_path / "raw" / doc.filename, [["1. Introduction", body, "Page 1"]])
    embeddings = TinyEmbeddings(tokenizer)
    first = build_index(settings, embedder=embeddings)
    second = build_index(settings, embedder=embeddings)
    assert first["reused"] is False and second["reused"] is True
    assert first["chunks"] == second["chunks"] == 2
    for _ in range(2):
        retriever = Retriever(settings, embedder=embeddings)
        try:
            assert retriever.search("confabulation")[0].chunk.document_id == "genai"
            assert retriever.search("GOVERN")[0].chunk.document_id == "rmf"
            assert retriever.search("  ") == []
        finally:
            retriever.close()
    forced = build_index(settings, force=True, embedder=embeddings)
    assert forced["chunks"] == 2 and not forced["reused"]


def test_failed_rebuild_preserves_existing_index(tmp_path, tokenizer):
    settings = Settings(data_dir=tmp_path)
    for doc in DOCUMENTS:
        make_pdf(
            tmp_path / "raw" / doc.filename,
            [
                [
                    "1. Introduction",
                    "GOVERN is one of the functions in the NIST AI Risk Management Framework.",
                ]
            ],
        )
    embeddings = TinyEmbeddings(tokenizer)
    build_index(settings, embedder=embeddings)
    before = (settings.index_dir / "manifest.json").read_bytes()

    class FailingEmbeddings(TinyEmbeddings):
        def documents(self, texts):
            raise RuntimeError("model failed")

    with pytest.raises(RuntimeError):
        build_index(settings, force=True, embedder=FailingEmbeddings(tokenizer))
    assert (settings.index_dir / "manifest.json").read_bytes() == before


def test_rejects_html_download_without_leaving_partial_file(tmp_path, monkeypatch):
    def fake_client(**kwargs):
        return original_client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, text="<html>Unavailable</html>")
            ),
            **kwargs,
        )

    original_client = httpx.Client
    monkeypatch.setattr("nist_rag.ingest.httpx.Client", fake_client)
    with pytest.raises(AppError, match="not a readable PDF"):
        download_pdf(DOCUMENTS[0], tmp_path)
    assert not (tmp_path / DOCUMENTS[0].filename).exists()
    assert not list(tmp_path.glob("*.part"))


def test_rrf_handles_incompatible_score_scales_without_counting_duplicates():
    a, b, c = [str(uuid5(NAMESPACE_URL, key)) for key in ["a", "b", "c"]]
    scores = fuse_ranks([[a, b], [b, c, b]])
    assert max(scores, key=scores.get) == b
    assert scores[b] == pytest.approx(1 / 62 + 1 / 61)


def test_missing_index_is_actionable(tmp_path, tokenizer):
    with pytest.raises(AppError, match="ingest --force"):
        Retriever(Settings(data_dir=tmp_path), embedder=TinyEmbeddings(tokenizer))
