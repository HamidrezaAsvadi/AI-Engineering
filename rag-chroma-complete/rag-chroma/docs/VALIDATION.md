# Validation record

Checked on September 28, 2026, with Python 3.12.14 on Linux.

- 15 pytest cases passed using actual temporary ChromaDB collections.
- Verified persistence on reopening, idempotent ingestion, document replacement,
  source filtering, deletion, empty-corpus abstention, citation label handling,
  context budgeting, and pipeline-fingerprint mismatch rejection.
- Simulated provider and storage failures to verify preservation of the old
  document and rollback of the attempted new version.
- Verified chunk progress/overlap, Unicode, page metadata, extraction errors,
  scanned/blank PDF rejection, and input validation.
- Exercised upload/chat/search validation, optional bearer authentication,
  deletion, upload size limits, and static frontend routes through TestClient.
- Tested actual OpenAI SDK request serialization and response parsing with an
  HTTP mock transport. No live model calls were made.
- Dependency consistency check (`pip check`) passed.
- Python compilation and JavaScript syntax checks passed.
- Started Uvicorn and checked real HTTP routes and the CLI against an empty database.

One upstream Starlette warning notes that its HTTPX TestClient integration is
being deprecated. It did not affect the passing tests in the pinned environment.

Not executed: live OpenAI embedding/generation calls, real semantic evaluation,
Docker build/runtime (Docker unavailable in the build environment), or visual
browser automation. The included evaluation commands must be run with a valid API
key to measure answer quality. No retrieval/answer-quality scores are claimed.

The dependency snapshot in requirements-lock.txt records this Linux environment.
The ordinary requirements.txt pins the tested direct application dependencies.
