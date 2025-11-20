# Graph-RAG Ophthalmology Assistant

## Overview

Text-Graph-RAG is a Graph Retrieval-Augmented Generation system tailored for ophthalmic diseases. It combines:

1. **Knowledge Graph Construction** – `ingest.py` relies on the lightweight `nano_graphrag` pipeline to chunk Markdown documents (e.g., `demo_data/eyes.md`), extract diseases / symptoms / treatments, and populate Neo4j (Disease, Symptom, Treatment, Medication, RiskFactor, CareTip).
2. **Hybrid Question Answering** – `main.py` orchestrates `GraphDataPreparationModule`, `MilvusIndexConstructionModule`, `HybridRetrievalModule`, `GraphRAGRetrieval`, `IntelligentQueryRouter`, and the OpenAI-powered `GenerationIntegrationModule` to answer user questions with graph-aware reasoning.

## Repository Layout

| Path | Description |
| --- | --- |
| `demo_data/` | Sample disease markdowns (with `## 别名 / alias` sections) |
| `rag_modules/` | Core Graph-RAG modules (data prep, indexing, retrieval, generation, router, ingestor) |
| `nano_graphrag/` | Lightweight GraphRAG utilities (chunking, LLM prompts, storage backends) |
| `docker-compose.yml` | Milvus single-node stack (etcd + MinIO + milvus-standalone) |
| `ingest.py` | CLI for Markdown → Graph → Neo4j ingestion |
| `main.py` | Starts the “BrightSight” disease assistant |

## Requirements

- Python ≥ 3.10 (conda environment `graph-rag` recommended)
- `pip install -r requirements.txt`
- Docker & Docker Compose (for Milvus standalone deployment)
- Neo4j (either via Docker or external server)
- OpenAI API key (`OPENAI_API_KEY`)

Typical environment variables:

```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=all-in-rag

export MILVUS_HOST=localhost
export MILVUS_PORT=19530

export OPENAI_API_KEY=sk-xxxx
```

## Quick Start

1. **Install dependencies**
   ```bash
   conda activate graph-rag
   pip install -r requirements.txt
   ```

2. **Start infrastructure**
   ```bash
   docker compose up -d
   docker compose ps
   ```
   - Neo4j Browser: `http://localhost:7474`
   - Milvus health check: `http://localhost:9091/healthz`

3. **Build the graph**
   ```bash
   # (Optional) reset Neo4j/Milvus before ingesting
   cypher-shell -u neo4j -p all-in-rag "MATCH (n) DETACH DELETE n;"
   python drop_milvus_collection.py

   python ingest.py \
     --data-path demo_data \
     --domain medical \
     --llm-concurrency 16 \
     --working-dir ./.nano_cache_medical
   ```

4. **Launch the assistant**
   ```bash
   python main.py
   ```
   CLI commands: `stats`, `rebuild`, `quit`, plus free-form questions (“眼睛刺痛怎么办?”).

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Fail connecting to server on localhost:19530` | Milvus containers are down. Run `docker compose up -d` and retry. |
| `field entity_name not exist` | Old Milvus schema still active. Drop the `cooking_knowledge` collection and re-run `ingest.py`. |
| OpenAI 401 / `Invalid Authentication` | Ensure `OPENAI_API_KEY` is exported; check proxy / network settings. |
| `Expecting value: line 1 column 1 (char 0)` | Some OpenAI responses include natural language before JSON. Add `response_format={"type": "json_object"}` or trim the prefix before `json.loads` in `IntelligentQueryRouter`. |

## Bonus Utilities

- `drop_milvus_collection.py` – small helper to remove the current Milvus collection.
- `agent/run_ai_agent.py` – the original recipe-ingestor workflow (handy if you want to bootstrap non-medical data).

## License & Usage

This repository is intended for research and prototyping. Make sure you have permission to use any medical content, and secure your OpenAI credentials before deploying to production environments.
