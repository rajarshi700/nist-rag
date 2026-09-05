# Verification notes

Prepared on September 5, 2026, using Python 3.12.13 on Linux.

## Results

- The project installed successfully as an editable Python package, including development dependencies.
- **33 automated tests passed.** These used the real LangGraph runtime and local Qdrant,
  with scripted LLM responses and mocked HTTP calls.
- Ruff passed. All application and evaluation modules compiled successfully.
- Both official NIST PDFs were downloaded through the application's downloader into a
  fresh directory. Their checksums matched the bundled copies.
- Real FastEmbed MiniLM inference produced 384-dimensional vectors. The final ingestion
  produced **303 chunks** from the two PDFs (48 and 64 physical pages).
  Covers, contents, and the GenAI bibliography were excluded from searchable content.
- Repeating ingestion reused the existing index. The installed `nist-rag` entry point worked.
- **6 of 6 real retrieval cases passed at top-6.** Four had supporting evidence
  at rank 1 and two at rank 2. The recorded result is in
  [examples/retrieval_report.json](examples/retrieval_report.json).
- Starting chat without an API key returned an actionable configuration error.

## What has not been verified

**No live Groq completion was run: no Groq API key was available in the build environment.**
The tests verify orchestration, conversation state, retry bounds, response parsing,
citation handling, and failure paths; they do not establish the real model's answer quality.
Model access, account quota, live latency, and end-to-end answer correctness remain to be
checked with your key. The retrieval cases are a small development set, not an independent
quality benchmark.

After adding your key to `.env`, run:

```bash
python -m scripts.evaluate --live --output live-evaluation.json
```

Review the actual answers and PDF citations. The script includes a first question, two
follow-ups, a topic change to confabulation, an unrelated question, and an unsupported
numerical premise. Its automatic live checks inspect result statuses; they cannot prove
semantic correctness.

## Tested package versions

| Package | Version |
| --- | --- |
| `langgraph` | `1.2.11` |
| `langchain-core` | `1.6.2` |
| `qdrant-client` | `1.19.0` |
| `fastembed` | `0.7.4` |
| `httpx` | `0.28.1` |
| `pypdf` | `6.17.0` |
| `pydantic` | `2.13.5` |
| `tokenizers` | `0.23.2` |
| `pytest` | `8.4.2` |

Direct application dependencies are pinned in `pyproject.toml`. Transitive dependencies
are resolved by pip. The ZIP does not include a virtual environment, embedding weights,
or a prebuilt database; `ingest` creates the local index.

## Bundled PDF checksums (SHA-256)

- `nist-ai-rmf-1.0.pdf`: `7576edb531d9848825814ee88e28b1795d3a84b435b4b797d3670eafdc4a89f1`
- `nist-genai-profile.pdf`: `6e73620ab6b64e90ef2c04bf0e0d6246185a2f4b1b13cab0df494496cff89b6a`

These are copies of the official NIST publications. No changes were made to the PDF files.
