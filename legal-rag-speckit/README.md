# Legal RAG PoC — Spec Kit

This directory is a complete Spec Kit artifact set for the Arabic Legal RAG freelance proof of concept.

## Artifact map

- `.specify/memory/constitution.md` — project governing principles
- `specs/001-legal-rag-poc/spec.md` — product/feature requirements
- `specs/001-legal-rag-poc/plan.md` — implementation architecture
- `specs/001-legal-rag-poc/research.md` — technical decisions
- `specs/001-legal-rag-poc/data-model.md` — data contracts/models
- `specs/001-legal-rag-poc/contracts/openapi.yaml` — logical API contract
- `specs/001-legal-rag-poc/quickstart.md` — local setup
- `specs/001-legal-rag-poc/tasks.md` — implementation tasks
- `specs/001-legal-rag-poc/checklist.md` — final review checklist

## Intended implementation

Initial knowledge base: bilingual Egyptian Civil Code.

Core stack:
- Python
- BGE-M3
- Qdrant
- BGE-reranker-v2-m3
- Redis semantic cache
- Streamlit
- configurable LLM provider

Explicitly excluded:
- notebooks
- MLflow
- DVC
- Langfuse
- Prometheus/Grafana
- BentoML
- production authentication
- production MLOps

## Source basis

The design uses the supplied ITI × MLOps MENA project handbook for the legal corpus requirements, including structured article-level JSON, Arabic/English handling, article-number normalization, repealed-article flags, metadata hierarchy, article-level chunking, and article citations.
