# NIST AI RMF Question Answering

A small conversational RAG application for NIST AI RMF 1.0 and its Generative AI Profile.
It uses Python, LangGraph, local FastEmbed embeddings, embedded Qdrant, and Groq.
The interface is a terminal chat so the assignment stays focused on retrieval and orchestration.

## Quick start

Python **3.12** is recommended. Run the commands from the extracted `nist-rag` folder.
No Docker, GPU, or database server is needed.

**macOS / Linux**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

On Windows, use `.\.venv\Scripts\python.exe` in place of `python` in the commands below.
This avoids needing to change PowerShell's script execution policy.

Open `.env` and add a [Groq API key](https://console.groq.com/keys):

```dotenv
GROQ_API_KEY=your_key_here
GROQ_MODEL=openai/gpt-oss-20b
```

Then run:

```bash
python -m nist_rag ingest
python -m nist_rag chat --trace
```

Ingestion downloads any missing PDFs, loads the embedding model, and saves the index.
The ZIP includes the official PDFs for convenience; they are reused when present.
The embedding weights are downloaded on first use and cached in `data/models/`.
**Ingestion and search need no LLM key. Chat needs a valid key and available API quota.**

Try this conversation in the same chat session:

```text
What are the four core functions of the NIST AI RMF?
Which one is cross-cutting?
How does it relate to the other functions?
```

Use `/new` to start a new conversation and `/quit` to exit. Memory lasts for the current
process. Separate `ask` invocations do not share history.

## Commands

```bash
# Build or reuse an index; --force rebuilds it from the local PDFs.
python -m nist_rag ingest
python -m nist_rag ingest --force

# One question with a fresh conversation.
python -m nist_rag ask "What is confabulation in the Generative AI Profile?" --trace

# Inspect the actual passages, ranks, and source metadata without using an LLM.
python -m nist_rag search "NIST data privacy risks"
python -m nist_rag search "GOVERN cross-cutting" --json

# Machine-readable answer, sources, graph path, and result status.
python -m nist_rag ask "What is the MAP function?" --json
```

The `search` command shows reciprocal-rank-fusion, cosine, and BM25 scores. They are
different measures, **not probabilities of correctness**. The LLM separately assesses
whether the retrieved text actually supports an answer.

## Documents and citations

| Document | Official source | Included filename |
| --- | --- | --- |
| NIST AI RMF 1.0, NIST AI 100-1 | [PDF](https://nvlpubs.nist.gov/nistpubs/ai/nist.ai.100-1.pdf) | `data/raw/nist-ai-rmf-1.0.pdf` |
| Generative AI Profile, NIST AI 600-1 | [Publication page](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence), [PDF](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf) | `data/raw/nist-genai-profile.pdf` |

To exercise the downloader, move the two bundled PDFs out of `data/raw/` and run `ingest`.
If a corporate network blocks NIST downloads, download the files through the official
links above, save them with these filenames, and rerun the same command.

Each chunk stores its document ID, title, PDF page, detected printed page, detected section,
source URL, and stable UUID. **PDF pages are one-based viewer pages**; they can differ
from the numbers printed on the document. Links use `#page=N` to open the physical page.

An illustrative answer format is:

```text
The four functions are Govern, Map, Measure, and Manage. [1]

Sources:
[1] NIST AI RMF 1.0 — Section 5. AI RMF Core
    — PDF page 25 (printed page 20)
    https://nvlpubs.nist.gov/nistpubs/ai/nist.ai.100-1.pdf#page=25
```

This is an example of expected formatting, not a recorded live LLM response.

## Design choices

- **Chunking:** page and heading boundaries first, then up to 220 WordPiece tokens with
  32 tokens of overlap. The splitter uses the embedding model's tokenizer and source-text
  offsets. It prefers nearby sentence or line endings. This keeps chunks below MiniLM's
  256-token training window and keeps page references unambiguous.
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` through FastEmbed produces
  384-dimensional vectors locally using ONNX. It is small enough for a CPU-only assignment.
- **Vector store:** Qdrant's local mode persists vectors without a server. The application
  rebuilds BM25 from the saved chunk text when it starts.
- **Retrieval:** retrieve 20 or more dense and lexical candidates, combine ranks with
  reciprocal rank fusion, remove near-duplicate passages, and pass the best six to the LLM.
  Generic corpus words such as “NIST” and “profile” are omitted from lexical queries when
  there are more specific terms. Dense retrieval still sees the complete question.
- **Corpus cleanup:** omit covers and contents pages, plus the GenAI bibliography.
  Page ranges are recorded for these two specific editions. The RMF body and appendices
  and the GenAI body and Appendix A remain searchable. Normalize ligatures and remove
  repeated page furniture. Section labels are detected conservatively and can be absent.
- **Graph:** LangGraph owns every decision about contextualization, retrieval, assessment,
  one retry, answering, and abstention. It is not a wrapper around a separate RAG chain.
- **Grounding:** the relevance check selects usable excerpts; the answer must cite those
  excerpts for every claim. Unknown source IDs are rejected. Titles, pages, and links are
  rendered by application code, never supplied by the answer model.

See [DESIGN.md](DESIGN.md) for the graph, state, and trade-offs.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `GROQ_API_KEY` | Empty | Required for chat and live evaluation only |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | Groq-hosted model supporting JSON object output |
| `DATA_DIR` | `./data` | PDFs, cached model, and local index |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model; rebuild after changing it |
| `TOP_K` | `6` | Evidence passages per retrieval, between 1 and 10 |
| `HTTP_TIMEOUT` | `45` | LLM HTTP timeout in seconds |
| `MAX_API_RETRIES` | `2` | Additional attempts for transient or malformed responses |
| `MAX_RETRY_WAIT` | `30` | Maximum automatic wait in seconds |

The current Groq free-plan documentation lists GPT-OSS 20B; model access and limits depend
on the account and can change. Check [models](https://console.groq.com/docs/models) and
[rate limits](https://console.groq.com/docs/rate-limits) if the API rejects a model.
For GPT-OSS, the client uses low reasoning effort and reserves an extra 1,024 completion
tokens for reasoning. Reasoning text is not requested or displayed.

API requests are synchronous. Rate-limit responses respect `Retry-After` when the requested
wait is within the configured limit. Longer waits are reported to the user without retrying
early. Timeouts and selected server errors use bounded retries. Invalid JSON and schema
violations also get bounded retries. Authentication and ordinary bad-request errors stop
immediately. A failed API call is reported as an error, not as “the documents have no answer.”

A normal first turn makes two logical LLM calls: assessment and answer. A follow-up adds
one contextualization call. A retrieval retry adds one assessment call; the rewrite uses
the query already proposed by the assessor. Transport/format retries can add API requests.

## Tests and evaluation

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check .

# Real document retrieval and local embeddings, no LLM key needed.
python -m scripts.evaluate

# Real model calls, including two follow-ups, a topic change, and abstention checks.
python -m scripts.evaluate --live --output live-evaluation.json
```

Unit and integration tests run offline with small synthetic PDFs, a real local Qdrant
store, the real LangGraph runtime, scripted LLM responses, and mocked HTTP responses.
They check three-turn memory, thread isolation, fresh retrieval on follow-ups, bounded
retry paths, abstention, invalid citations, index rebuilds, and API failures.

The six retrieval cases use the actual NIST PDFs and actual MiniLM embeddings. Their target
pages are in `examples/retrieval_cases.json`; the check requires both the intended page and
supporting terms. This is a small development sanity check, not a held-out benchmark.

Live evaluation compares response statuses and saves the full answers and citations. Read
those answers against their cited pages to assess semantic accuracy. It does not claim to
automatically prove grounding. See [VERIFICATION.md](VERIFICATION.md) for what was run when
this ZIP was prepared.

### Verification results

Results recorded on September 5, 2026:

| Check | Result |
| --- | --- |
| Automated tests | 33 passed |
| Ruff linting | Passed |
| Retrieval evaluation | 6/6 passed at top-6 |
| Conversational follow-ups | 2/2 passed |
| Live Groq evaluation | 6/6 expected outcomes passed |
| Failure handling | 2/2 questions safely abstained |

The retrieval cases are a small development sanity check, not an
independent benchmark. Detailed results are available in
`examples/retrieval_report.json` and `live-evaluation.json`.

## More questions to try

- What are the characteristics of trustworthy AI?
- What does the MAP function do?
- What is confabulation, and why is it a risk?
- What privacy risks does the Generative AI Profile describe?
- How does the Generative AI Profile relate to AI RMF 1.0?
- What risks come from value chain and component integration?
- Who won yesterday's football match? — should be outside the knowledge base.
- What exact dollar fine does NIST require for every AI RMF violation? — should not
  invent a fine or accept the question's premise.

## Known limitations

- The relevance and answer models can still make mistakes. Valid citation IDs do not
  prove that every sentence is entailed by a passage; semantic grounding needs review.
- PDF text extraction can flatten table columns and miss text in figures. There is no OCR.
  Heading detection is heuristic; physical page links are the reliable citation anchor.
- Small chunks can separate table headers from rows or split a longer explanation. Six
  passages and one retry are deliberately bounded, so broad summaries may be refused.
- Memory is process-local. Only the last three user/assistant turns are passed to the
  follow-up rewriter; checkpoints still retain the full session until the process exits.
- Embedded Qdrant supports one application process per data directory here. Close chat
  before rebuilding or running evaluation against the same directory.
- This is a CLI prototype. It has no login, web UI, background jobs, streaming, or deployment setup.
- The first model download requires internet access; live answers require Groq access.
  Never commit a real `.env` or put an API key in the source code.

## Project files

| Path | Purpose |
| --- | --- |
| `nist_rag/ingest.py` | Download, PDF cleanup, chunking, and index creation |
| `nist_rag/embeddings.py` | Local embedding model and tokenizer |
| `nist_rag/retrieval.py` | Dense search, BM25, and rank fusion |
| `nist_rag/graph.py` | LangGraph state, nodes, routes, citations, and session wrapper |
| `nist_rag/llm.py` | Groq HTTP client, schema validation, and retries |
| `nist_rag/prompts.py` | Contextualization, evidence assessment, and answer instructions |
| `nist_rag/cli.py` | Terminal chat and utility commands |
| `tests/` | Offline behavioral and integration tests |
| `scripts/evaluate.py` | Real retrieval checks and optional live conversation checks |

## References

- [LangGraph graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)
- [LangGraph memory](https://docs.langchain.com/oss/python/langgraph/add-memory)
- [Qdrant client local mode](https://github.com/qdrant/qdrant-client#local-mode)
- [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [FastEmbed models](https://qdrant.github.io/fastembed/examples/Supported_Models/)
- [Groq API reference](https://console.groq.com/docs/api-reference)
- [Groq JSON output](https://console.groq.com/docs/structured-outputs)
