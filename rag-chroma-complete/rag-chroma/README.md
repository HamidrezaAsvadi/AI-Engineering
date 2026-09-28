# Chroma RAG Knowledge Assistant

A complete local learning project: upload documents, store embeddings in ChromaDB,
retrieve relevant passages, and generate answers with source citations. Includes a
browser interface, FastAPI API, CLI, sample documents, evaluation, tests, and Docker.

**Start here:** follow the quick start below, then read [A-to-Z walkthrough](docs/A_TO_Z.md).
This is a single-user, single-process baseline you can understand and extend. It is
not an enterprise deployment. Every file in `data/documents` is fictional demo data.

## Stack

| Layer | Implementation | What it does |
|---|---|---|
| Language | Python 3.12 | Application and pipeline |
| Extraction | pypdf + UTF-8 text reader | PDF, TXT, Markdown |
| Chunking | Explicit Python splitter | Page-aware overlapping character chunks |
| Embeddings | OpenAI `text-embedding-3-small` | Converts chunks and queries into vectors |
| Vector database | ChromaDB `PersistentClient` | Local persistent vectors, text, metadata |
| Retrieval | Chroma cosine nearest-neighbor search | Top-K passages, optional source filter/cutoff |
| Generation | OpenAI Responses API, `gpt-4.1-mini` | Evidence-based answers with citation labels |
| Backend | FastAPI + Uvicorn | Upload, search, ask, list, delete |
| Frontend | HTML, CSS, JavaScript | Upload files and inspect answers/passages |
| Validation | pytest + HTTPX | Offline tests with actual Chroma storage |
| Packaging | Docker + Compose | Repeatable local launch and persistent volume |

Both model names are configurable. Use a compatible model available to your account.
The app calls OpenAI directly; it does not require LangChain or LlamaIndex.

## 1. Local quick start

You need Python 3.12, an OpenAI API key with access to the configured models, and
network access to install packages and call the API. The actual RAG path requires
API credits; offline tests require no key. Never put your key in browser code.

Extract the ZIP and open a terminal in the `rag-chroma` directory.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

Set `OPENAI_API_KEY=your_actual_key` in `.env`, save, then start the API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Open a **second PowerShell terminal** in the same project directory:

```powershell
.\.venv\Scripts\python.exe -m app.cli ingest data/documents
.\.venv\Scripts\python.exe -m app.cli ask "How long does the gateway buffer telemetry when offline?"
```

No PowerShell activation-policy change is needed because these commands use the
virtual environment's interpreter directly. For the later `python ...` commands,
use `.\.venv\Scripts\python.exe ...` on Windows.

### macOS or Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` to set `OPENAI_API_KEY`, then:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

In a second terminal, enter the same project directory and activate `.venv`:

```bash
source .venv/bin/activate
python -m app.cli ingest data/documents
python -m app.cli ask "How long does the gateway buffer telemetry when offline?"
```

### Open the application

- Browser interface: <http://127.0.0.1:8000>
- Interactive API documentation: <http://127.0.0.1:8000/docs>
- Liveness check: <http://127.0.0.1:8000/health>

A typical answer to the sample question should say **up to 24 hours**, with a
citation such as `[S1]`. Wording and retrieval order may vary. Expand the source
card and check `gateway.md` yourself. This is an expected example, not a recorded
live OpenAI result.

Try these next:

1. “What happens to invalid telemetry messages?”
2. “How are telemetry duplicates identified?”
3. “How is the idle-detection threshold chosen?”
4. “What is the annual subscription price?” — the documents do not provide it.

## 2. Add your documents

Use the browser upload control or the CLI:

```bash
python -m app.cli ingest path/to/manual.pdf
python -m app.cli ingest path/to/documents
python -m app.cli list
python -m app.cli ask "What is the retention period?" --source support.md
python -m app.cli ask "MQTT gateway" --search-only --top-k 3
python -m app.cli delete support.md
```

Directory ingestion is recursive. Duplicate basenames in a directory are rejected
before upload: rename them so every source has a unique name. Document identity
is the filename, including case. Uploading changed content under the same filename
replaces its chunks. Uploading identical bytes under that name skips embedding.
Deleting a local file does not delete its indexed copy; use `delete` explicitly.
Original file bytes are not retained by the app; keep your own source documents.

Supported: text PDFs, UTF-8 TXT and Markdown. Scanned PDFs need OCR first.
Limits: 10 MiB upload, 500 PDF pages, 2 million extracted characters per document.
A mixed PDF with some scanned pages can index the text pages while omitting the
image-only pages. Complex tables/columns may extract poorly; inspect the passages.

## 3. Configuration

All settings are read from `.env` in the current working directory. Restart the
API after changes. Empty optional values use defaults.

| Setting | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | Empty | Server-side provider key; required for embeddings/generation |
| `CHAT_MODEL` | `gpt-4.1-mini` | Responses API model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Shared document/query embedding model |
| `CHROMA_PATH` | `./storage/chroma` | Database directory |
| `COLLECTION_NAME` | `knowledge_base` | Chroma collection |
| `CHUNK_SIZE` | `1600` | Maximum characters per chunk, not tokens |
| `CHUNK_OVERLAP` | `200` | Character overlap; must be less than chunk size |
| `TOP_K` | `5` | Default retrieved chunk count for API requests |
| `MAX_DISTANCE` | Empty | Optional cosine-distance cutoff; lower is closer |
| `CONTEXT_MAX_CHARS` | `12000` | Evidence text budget, excluding JSON/prompt overhead |
| `MAX_OUTPUT_TOKENS` | `1200` | Response output-token budget |
| `APP_API_KEY` | Empty | Optional shared API bearer token |
| `RAG_API_URL` | `http://127.0.0.1:8000` | CLI/evaluator API URL |

The UI and CLI explicitly default to Top-K 5; use their controls/flags to change
it. The retrieval evaluator defaults to 3. A distance is not a confidence
probability. `MAX_DISTANCE` is disabled initially so an arbitrary cutoff does
not hide useful evidence; evaluate and calibrate before enabling one.

Changing the embedding model or chunk settings requires a new collection name
and re-ingestion. A pipeline fingerprint prevents silent mixing. Changing only
the chat model does not require re-embedding. Keep the same embedding model for
queries and documents even if two models happen to have the same dimensions.

## 4. API usage

```bash
curl -X POST http://127.0.0.1:8000/api/documents \
  -F 'file=@data/documents/gateway.md'

curl -X POST http://127.0.0.1:8000/api/search \
  -H 'Content-Type: application/json' \
  -d '{"question":"What happens during a network outage?","top_k":3}'

curl -X POST http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"How long does offline buffering last?","source":"gateway.md"}'
```

On Windows use `curl.exe` with shell-appropriate quoting, or use the Python CLI.
If `APP_API_KEY` is set, send `Authorization: Bearer <your-app-token>`; enter that
same token in the browser field. Do **not** enter the OpenAI key there. CLI and
evaluation read the app token from `.env`. This token grants access to the whole
knowledge base and deletion; it is not a user/role permission system.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Process liveness, not provider readiness |
| GET | `/api/documents` | List indexed filenames and chunk counts |
| POST | `/api/documents` | Multipart upload and index |
| DELETE | `/api/documents` | Delete by JSON body `{"source":"gateway.md"}` |
| POST | `/api/search` | Retrieve passages without generation |
| POST | `/api/chat` | Retrieve and generate a cited answer |

Chat returns `answer`, `sources`, `cited_labels`, `warnings`, and `latency_ms`.
Every source contains text, source filename, 1-based PDF page (0 for TXT/MD),
chunk index, ID, label, and cosine distance. Sources are the exact evidence sent
to the LLM; `cited_labels` identifies which of them the answer cites. Citation
label validity does not prove that the cited passage supports the claim.

## 5. Evaluation and tests

Start the API and ingest the four sample documents first:

```bash
python -m evaluation.run --top-k 3
python -m evaluation.run --top-k 3 --generate --output evaluation/results-generation.json
```

Retrieval-only evaluation still calls the embedding API. `--generate` adds LLM
calls. Reports include per-question results, document-level Hit@K, source Recall@K,
MRR, and latency. Generation adds exact-phrase abstention checks and expected-term
coverage. These are diagnostic smoke metrics, not a faithfulness guarantee.

The tiny dataset includes multi-source and unanswerable questions. The candidate
budget K counts chunks; metric ranking deduplicates filenames. Unanswerable items
are excluded from recall/MRR and evaluated for abstention with `--generate`.
Create a held-out dataset from your actual domain before claiming quality.

Run offline tests (no API key or live model requests):

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

See [VALIDATION.md](docs/VALIDATION.md) for what was actually checked during
project creation. `requirements.txt` pins direct dependencies. The full
`requirements-lock.txt` records the tested Linux/Python 3.12 environment,
including development and platform-specific transitive packages. Use the normal
requirements on Windows/macOS; the Linux snapshot is not a universal lock file.

## 6. Docker

Copy `.env.example` to `.env`, add your key, and run:

```bash
docker compose up --build -d
docker compose exec rag python -m app.cli ingest data/documents
docker compose exec rag python -m app.cli ask "How long does offline buffering last?"
```

Open <http://127.0.0.1:8000>. The named volume `rag-storage` persists Chroma data.

```bash
docker compose logs -f rag
docker compose down
```

Stopping the container keeps the volume. `docker compose down -v` would delete
the stored database, so use it only for an intentional reset. For backups, stop
writes/the application before copying the persistent database directory or volume.
The container runs as a non-root user and binds the host port to loopback.

## 7. Read the code in this order

| File | Responsibility |
|---|---|
| `app/config.py` | Typed environment settings and validation |
| `app/documents.py` | Extract, normalize, split, identify documents |
| `app/provider.py` | Batched embeddings and Responses API generation |
| `app/rag.py` | Chroma storage, replacement, retrieval, context, citations |
| `app/main.py` | HTTP endpoints, authentication, error handling |
| `app/static/` | Browser upload, question, answer and source display |
| `app/cli.py` | HTTP client for ingestion and questions |
| `evaluation/run.py` | Labelled retrieval/generation smoke evaluation |
| `tests/test_rag.py` | Offline behavior and SDK-contract checks |
| `docs/A_TO_Z.md` | Concepts, design choices, debugging, production extensions |

## 8. Troubleshooting

| Symptom | Action |
|---|---|
| `Set OPENAI_API_KEY` | Edit `.env` in the project directory and restart |
| Provider request failed | Check API credentials, account model access, credits, rate limits, network |
| Cannot reach API | Start Uvicorn first; check port and `RAG_API_URL` |
| Empty answers/no sources | List documents; inspect `/api/search`; check source filter and cutoff |
| Wrong evidence | Inspect extracted text, improve chunking, tune K, add better domain examples |
| Correct evidence, wrong answer | Improve prompting/model and evaluate citation support |
| Pipeline configuration changed | Set a new `COLLECTION_NAME`, restart, re-ingest |
| No extractable PDF text | OCR the PDF externally, then ingest the resulting text/PDF |
| Chroma locked or inconsistent | Use ONE API process and do not open the same directory from another process |
| Port 8000 in use | Stop the previous instance or change Uvicorn port and `RAG_API_URL` |

## Scope and data flow

Documents and questions are sent to OpenAI for embeddings; retrieved passages and
questions are sent for generation. `store=False` is set on Responses requests;
it is not a blanket zero-retention guarantee. Keep private data handling aligned
with your provider/account policies. Chroma stores extracted text locally, not
only vectors. `.env`, database files, and caches are excluded from version control.

This app uses one shared collection, no conversational memory, no OCR, no neural
reranker, no hybrid keyword search, and no tenant-level access control. The UI
answers independent questions. Keep it on localhost while learning. The walkthrough
explains concrete changes required for a multi-user deployment.

## Official references

Implementation reference pages checked September 28, 2026:

- [Chroma collection configuration](https://docs.trychroma.com/docs/collections/configure)
- [Chroma query and get](https://docs.trychroma.com/docs/querying-collections/query-and-get)
- [OpenAI embeddings](https://developers.openai.com/api/docs/guides/embeddings)
- [OpenAI text generation](https://developers.openai.com/api/docs/guides/text)
- [GPT-4.1 mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini)
- [FastAPI file uploads](https://fastapi.tiangolo.com/tutorial/request-files/)
