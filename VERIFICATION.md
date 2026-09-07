# Verification notes

Offline verification was initially performed on September 5, 2026, using Python 3.12.13 on Linux. Regression tests and live Groq evaluation were subsequently run on September 7, 2026, using Python 3.12 on Windows.

## Results

* The project installed successfully as an editable Python package, including development dependencies.
* **33 automated tests passed.** These used the real LangGraph runtime and local Qdrant with scripted LLM responses and mocked HTTP calls.
* Both official NIST PDFs were downloaded through the application's downloader into a fresh directory. Their checksums matched the bundled copies.
* Real FastEmbed MiniLM inference produced 384-dimensional vectors.
* Ingestion produced **303 chunks** from the two PDFs containing 48 and 64 physical pages.
* Covers, contents pages, and the Generative AI Profile bibliography were excluded from searchable content.
* Repeated ingestion correctly reused the existing index.
* The installed `nist-rag` command-line entry point worked.
* **6 of 6 real-document retrieval cases passed at top-6.** Four returned supporting evidence at rank 1 and two at rank 2.
* **6 of 6 live evaluation outcomes passed.**
* The live evaluation produced four grounded answers and two safe abstentions.
* Two dependent conversational follow-up questions were interpreted and answered correctly.
* Starting chat without an API key returned an actionable configuration error.

The retrieval report is available in [examples/retrieval_report.json](examples/retrieval_report.json). The complete live model responses, graph paths, rewritten queries, statuses, and citations are available in [live-evaluation.json](live-evaluation.json).

## Live evaluation

The live evaluation covered:

1. A question about the four AI RMF core functions.
2. A follow-up asking which function is cross-cutting.
3. A second follow-up asking how that function relates to the others.
4. A topic change to confabulation in the Generative AI Profile.
5. An unrelated football question.
6. An unsupported question about a required monetary fine.

The four document-related questions produced grounded answers with citations. The unrelated football question and unsupported fine question both produced safe abstentions without sources.

The football question returned `insufficient_context` instead of the more specific `out_of_scope` status. The evaluator accepts either status as a safe abstention because neither produces an unsupported answer or citation.

The evaluation completed with exit code `0`.

To reproduce it after configuring `GROQ_API_KEY` in `.env`, run:

```bash
python -m scripts.evaluate --live --output live-evaluation.json
```

## Evaluation scope

The automated live checks validate expected outcomes, follow-up resolution, abstention behaviour, and citation structure. They do not prove that every generated claim is semantically supported by its cited passage. The recorded answers and linked PDF pages should therefore also be reviewed manually.

The six retrieval cases are a small development sanity check, not an independent or held-out benchmark. Live model wording, latency, and rate-limit behaviour can vary between runs.

## Tested package versions

| Package          | Version  |
| ---------------- | -------- |
| `langgraph`      | `1.2.11` |
| `langchain-core` | `1.6.2`  |
| `qdrant-client`  | `1.19.0` |
| `fastembed`      | `0.7.4`  |
| `httpx`          | `0.28.1` |
| `pypdf`          | `6.17.0` |
| `pydantic`       | `2.13.5` |
| `tokenizers`     | `0.23.2` |
| `pytest`         | `8.4.2`  |

Direct application dependencies are pinned in `pyproject.toml`. Transitive dependencies are resolved by pip.

The repository does not include a virtual environment, embedding-model weights, or a prebuilt vector database. Running `ingest` creates the local index.

## Bundled PDF checksums

* `nist-ai-rmf-1.0.pdf`: `7576edb531d9848825814ee88e28b1795d3a84b435b4b797d3670eafdc4a89f1`
* `nist-genai-profile.pdf`: `6e73620ab6b64e90ef2c04bf0e0d6246185a2f4b1b13cab0df494496cff89b6a`

These files are unchanged copies of the official NIST publications.
