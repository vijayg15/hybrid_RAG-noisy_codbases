# Production-Ready Hybrid RAG for noisy Codebase

A modular implementation: ingest remote repositories, create AST-aware code chunks, run dense + sparse hybrid retrieval, expand through symbol dependencies, rerank candidates, generate grounded answers, and evaluate context precision/recall.

![Architecture](docs/architecture.png)

## Features

- GitHub, GitLab, and Bitbucket cloning over HTTPS; optional private-repository token.
- Provenance stored for repository, commit, file path, line range, symbol, and chunk ID.
- Python AST chunking plus Tree-sitter parsing for common languages; safe fallback windows for unsupported or malformed files.
- Dense vector search using Sentence Transformers and local Qdrant.
- Sparse BM25 search with reciprocal-rank fusion.
- Dependency-aware expansion through imports, symbol references, and function calls.
- Cross-encoder reranking before context-budget enforcement.
- OpenAI answer generation with strict source grounding; retrieval-only fallback when no key is configured.
- FastAPI endpoints: `POST /ingest`, `POST /query`, `GET|POST /eval`.
- Evaluation framework DeepEval for Contextual Precision and Contextual Recall.

## Project structure

```text
app/
├── api/routes.py              # FastAPI routes
├── core/config.py             # environment configuration
├── domain/models.py           # internal dataclasses
├── domain/schemas.py          # request/response models
├── services/
│   ├── repository.py          # clone/pull Git repositories
│   ├── chunking.py            # AST and Tree-sitter chunking
│   ├── indexing.py            # embeddings + Qdrant
│   ├── retrieval.py           # BM25, dense fusion, dependency graph, reranking
│   ├── generation.py          # context budget + grounded LLM answer
│   ├── evaluation.py          # precision/recall evaluation
│   └── pipeline.py            # orchestration
├── storage/
│   ├── database.py            # SQLAlchemy metadata store
│   └── chunk_store.py         # chunk persistence
└── main.py                    # application entrypoint

data/eval/sample_qrels.jsonl   # evaluation dataset template
docs/architecture.png          # architecture diagram
tests/                         # unit tests
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app
```

Swagger UI: `http://localhost:8000/docs`

The first startup downloads the embedding and reranking models. Set `OPENAI_API_KEY` in `.env` for generated answers. Without it, `/query` still returns the strongest grounded source matches and citations.

## API examples

### Ingest a public repository

```bash
curl -X POST http://localhost:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{
    "repo_url": "https://github.com/tiangolo/fastapi.git",
    "branch": "master"
  }'
```

### Ingest a private repository

```bash
curl -X POST http://localhost:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{
    "repo_url": "https://gitlab.com/acme/private-service.git",
    "token": "YOUR_TOKEN",
    "branch": "main"
  }'
```

The token is used only to authenticate the clone/fetch operation. The implementation resets the Git remote to the original non-token URL afterward.

### Query

```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "How is authentication middleware initialized?",
    "repo_id": "REPO_ID_FROM_INGEST",
    "top_k": 8
  }'
```

Every response contains source citations with `file_path`, `start_line`, `end_line`, `chunk_id`, and `symbol_name`.

### Evaluate

```bash
curl -X POST http://localhost:8000/eval \
  -H 'Content-Type: application/json' \
  -d '{
    "dataset_path": "data/eval/my_qrels.jsonl",
    "repo_id": "REPO_ID_FROM_INGEST",
    "k": 8
  }'
```

## Evaluation framework

Create JSONL records after ingestion, using actual chunk IDs returned by the index/database:

```json
{"question":"Where is an invoice persisted after charging?","repo_id":"abc123","relevant_chunk_ids":["chunk-a","chunk-b"]}
```

For each question:

- `Context Precision@K = relevant retrieved chunks / retrieved chunks`
- `Context Recall@K = relevant retrieved chunks / all annotated relevant chunks`

The report contains per-question scores and macro means. For credible evaluation, annotate 20–50 representative questions across navigation, implementation details, cross-file dependencies, configuration, and failure-handling behavior. Keep annotators blind to retrieval ranking where possible.

This ablation isolates the contribution of each retrieval stage. Record latency, Precision@K, Recall@K, and answer citation correctness.

## Design decisions

1. **AST-first chunking** preserves functions/classes and therefore improves source usefulness over fixed windows.
2. **Hybrid retrieval** covers semantic questions and exact identifiers.
3. **Reciprocal-rank fusion** avoids assuming dense and BM25 scores are calibrated.
4. **Dependency expansion** adds chunks connected through imports and referenced symbols.
5. **Cross-encoder reranking** applies expensive query-document scoring only to a bounded candidate pool.
6. **Stable provenance** makes every answer auditable to file, lines, and chunk.
7. **Local defaults** keep the demo easy to run; Qdrant, SQL metadata, and model providers can be swapped behind service boundaries.

## Production hardening

For a high-volume deployment, replace local Qdrant and SQLite with managed Qdrant/OpenSearch and PostgreSQL; enqueue ingestion with Celery, Dramatiq, or a cloud queue; encrypt repository tokens in a secret manager; add tenant isolation, webhooks for incremental indexing, observability, rate limiting, malware/file-size checks, and CI security scanning.

## Tests

```bash
pytest -q
```

## Docker

```bash
cp .env.example .env
docker compose up --build
```
