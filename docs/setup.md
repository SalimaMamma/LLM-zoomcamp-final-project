# Detailed setup

Back to the [README](../README.md). This document covers the full
installation, every environment variable, and the problems actually
encountered while building this project (not theoretical ones).

## Prerequisites

- Docker + Docker Compose (v2, `docker compose`, not `docker-compose`)
- A free Groq API key: https://console.groq.com (see
  [Groq free-tier quota](#groq-free-tier-quota) below)
- ~2GB disk space (Docker image + local `bge-small-en-v1.5` embedding
  model, downloaded on first `build_index.py` run)

Nothing else is required on the host machine: Postgres, Qdrant, Grafana
and the app all run in containers.

## Steps

```bash
git clone <this-repo-url>
cd sci-fitness-rag

cp .env.example .env
# edit .env: set GROQ_API_KEY at minimum

docker compose up -d          # Postgres, Qdrant, Grafana, app (Streamlit)
```

Then, in order (each step depends on the previous one):

```bash
# 1. Ingest scientific papers (Europe PMC, strict biomedical focus)
docker compose exec app python src/ingestion/europepmc.py
# Alternative/complement: broader, all-discipline coverage
docker compose exec app python src/ingestion/openalex.py

# 2. Chunking + vector index (Qdrant) + BM25 index
docker compose exec app python src/retrieval/build_index.py

# 3. Build the knowledge graph (uses Groq quota)
docker compose exec app python src/graphrag/extract_entities.py
```

The app is already running via `docker compose up -d` (it runs
continuously — `streamlit run` is the `app` container's default command):

- **App**: http://localhost:8502
- **Grafana**: http://localhost:3001 (login `admin` / `admin`)

## Environment variables

All defined in `.env` (see `.env.example` for defaults):

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | **Yes** | Groq API key (answer generation + graph extraction). Free at console.groq.com. |
| `GROQ_MODEL` | No | Which Groq model to use (default: `llama-3.3-70b-versatile`). |
| `OPENALEX_MAILTO` | No | Your email, to join OpenAlex's "polite pool" (much higher rate limit). No key required. |
| `SEMANTIC_SCHOLAR_API_KEY` | No | Optional, alternative ingestion source, not recommended (very strict rate limit without a key). |
| `POSTGRES_HOST` / `PORT` / `DB` / `USER` / `PASSWORD` | No | Defaults already match `docker-compose.yml`. Only change if you modify the compose file. |
| `QDRANT_HOST` / `PORT` / `COLLECTION` | No | Same. |
| `EMBEDDING_MODEL` | No | Local `sentence-transformers` embedding model (default: `BAAI/bge-small-en-v1.5`, CPU, free). |

Inside the containers, `docker-compose.yml` overrides
`POSTGRES_HOST=postgres` and `QDRANT_HOST=qdrant` (Docker-internal DNS
resolution) — the `localhost` values in `.env` are for running a script
outside a container (e.g. from a local Python venv on your machine, in
which case the host-mapped ports make those services reachable).

## Exposed ports

| Service | Host port | Notes |
|---|---|---|
| App (Streamlit) | `8502` | Remapped from Streamlit's default 8501 — see [why](#port-collisions) |
| Grafana | `3001` | Remapped from Grafana's default 3000 — same reason |
| Postgres | `5432` | Default |
| Qdrant | `6333` (REST), `6334` (gRPC) | Default |

## Troubleshooting



### Groq free-tier quota

The Groq free tier caps usage at **200,000 tokens/day** (across all
models/calls on the account). Graph extraction (`extract_entities.py`,
~1-2 calls per paper) and `llm_eval.py` (generation + judging, 2 calls per
question × strategy) are the biggest consumers. If you see:

```
groq.RateLimitError: Error code: 429 ... rate_limit_exceeded
```

## Reproducibility

- All dependency versions are pinned in `requirements.txt`.
- The corpus isn't a static versioned file: it's rebuilt on demand via the
  ingestion scripts (public APIs, no key required for Europe PMC/OpenAlex)
  — reproducible by anyone, though the exact contents may drift slightly
  over time as new papers get published/indexed.
- `data/eval_results/*.json` (real evaluation results) are committed to
  the repo so reviewers can see the scores without re-running everything.
- `data/*.pkl` (BM25 index) is not committed: it's regenerable in one
  command (`build_index.py`), and committing it would bloat the repo for
  no benefit.
