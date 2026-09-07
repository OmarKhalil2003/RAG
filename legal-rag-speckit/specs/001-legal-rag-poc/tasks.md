# Tasks: Arabic Legal RAG PoC

Input: `specs/001-legal-rag-poc/spec.md`, `plan.md`, `research.md`, `data-model.md`, and `contracts/openapi.yaml`.

Format: `[ID] [P?] [Story] Description`

- `[P]` = can run in parallel.
- `[Story]` maps implementation to a user story.

## Phase 1 — Setup

- [ ] T001 Create the repository structure defined in `plan.md`.
- [ ] T002 Initialize `pyproject.toml` with Python 3.11+ dependencies.
- [ ] T003 [P] Add `.env.example` and configuration loader in `src/legal_rag/config.py`.
- [ ] T004 [P] Add `.gitignore` excluding `.env`, model caches, Qdrant/Redis local data, virtual environments, and generated secrets.
- [ ] T005 [P] Add base Pydantic/domain models in `src/legal_rag/models.py`.
- [ ] T006 Add Docker Compose services for Qdrant and Redis in `docker-compose.yml`.

## Phase 2 — Corpus Ingestion

- [ ] T007 [P] Implement Arabic numeral normalization in `src/legal_rag/ingestion/normalizer.py`.
- [ ] T008 [P] Implement Arabic Unicode/orthographic normalization for retrieval text while preserving source text.
- [ ] T009 Implement bilingual PDF extraction in `src/legal_rag/ingestion/parser.py`.
- [ ] T010 Implement article boundary detection and hierarchy extraction.
- [ ] T011 Implement repealed-article detection/flagging.
- [ ] T012 Implement schema validation in `src/legal_rag/ingestion/validator.py`.
- [ ] T013 Implement `scripts/ingest.py`.
- [ ] T014 Add representative extraction fixtures and tests in `tests/test_parser.py`.
- [ ] T015 Add validation tests for Arabic-Indic article numbers, missing text, malformed article numbers, and repealed flags.

## Phase 3 — Vector Index

- [ ] T016 Implement BGE-M3 embedding adapter.
- [ ] T017 Implement Qdrant collection creation and payload schema in `src/legal_rag/retrieval/qdrant.py`.
- [ ] T018 Implement article-level dense indexing.
- [ ] T019 Implement sparse indexing using the selected BGE-M3/Qdrant-supported mechanism.
- [ ] T020 Implement `scripts/build_index.py`.
- [ ] T021 Add index smoke test against a small fixture corpus.

## Phase 4 — Query Analysis

- [ ] T022 [P] Implement language detection/heuristic in `src/legal_rag/retrieval/query_parser.py`.
- [ ] T023 [P] Implement article-number extraction for Western and Arabic-Indic numerals.
- [ ] T024 [P] Implement query normalization.
- [ ] T025 Implement `QuerySignals` construction.
- [ ] T026 Add tests for `Article 147`, `article 147`, `المادة 147`, and `المادة ١٤٧`.

## Phase 5 — Retrieval

- [ ] T027 Implement metadata filters for jurisdiction, law type, document ID, and repeal status.
- [ ] T028 Implement deterministic exact article retrieval.
- [ ] T029 Implement dense retrieval.
- [ ] T030 Implement sparse retrieval.
- [ ] T031 Implement reciprocal-rank fusion or equivalent candidate fusion.
- [ ] T032 Implement candidate deduplication by article number.
- [ ] T033 Implement BGE multilingual reranker adapter.
- [ ] T034 Implement final relevance threshold handling.
- [ ] T035 Add retrieval tests for exact article, semantic Arabic, semantic English, and cross-language queries.
- [ ] T036 Add tests proving law-type and jurisdiction filters prevent cross-scope retrieval.

## Phase 6 — Generation

- [ ] T037 Implement provider-agnostic LLM interface in `src/legal_rag/generation/llm.py`.
- [ ] T038 Implement grounded system prompt in `src/legal_rag/generation/prompts.py`.
- [ ] T039 Implement role-specific prompt variants.
- [ ] T040 Implement context builder that includes article number, legal hierarchy, Arabic text, English text, and repeal status.
- [ ] T041 Implement citation builder.
- [ ] T042 Implement grounded refusal when context is insufficient.
- [ ] T043 Add generation tests for source citation presence and out-of-scope refusal.

## Phase 7 — Semantic Cache

- [ ] T044 Implement Redis semantic-cache client in `src/legal_rag/cache/semantic_cache.py`.
- [ ] T045 Define cache namespace and metadata isolation.
- [ ] T046 Implement cache lookup using normalized query plus configured semantic similarity threshold.
- [ ] T047 Implement cache write with TTL.
- [ ] T048 Add knowledge-base and prompt version fields to cache metadata.
- [ ] T049 Add tests proving cache isolation across law type, jurisdiction, KB version, and prompt version.
- [ ] T050 Add cache hit/miss behavior to the application service.

## Phase 8 — RAG Orchestration

- [ ] T051 Implement `RAGService` in `src/legal_rag/pipeline.py`.
- [ ] T052 Implement execution order: query analysis → cache → exact/dense/sparse retrieval → fusion → reranking → context validation → generation → cache write.
- [ ] T053 Ensure Qdrant failure never falls back to unconstrained LLM generation.
- [ ] T054 Ensure Redis failure can degrade to uncached RAG without changing grounding behavior.
- [ ] T055 Add end-to-end tests with a small fixture corpus.

## Phase 9 — Streamlit UI

- [ ] T056 Implement Streamlit application shell in `app/streamlit_app.py`.
- [ ] T057 Add role selector.
- [ ] T058 Add jurisdiction and law-type selectors.
- [ ] T059 Add question input and submit action.
- [ ] T060 Display answer and article citations.
- [ ] T061 Display cache hit/miss status.
- [ ] T062 Display a concise source section without exposing internal vector IDs.

## Phase 10 — Evaluation

- [ ] T063 Create `evaluation/questions.json` containing exact article, Arabic, English, cross-language, repealed, metadata-filter, and out-of-scope cases.
- [ ] T064 Add at least 20 manually curated evaluation questions.
- [ ] T065 Include equivalent Arabic/English questions for the same legal concept.
- [ ] T066 Include Arabic-Indic article-number cases.
- [ ] T067 Include repealed article cases.
- [ ] T068 Include wrong-jurisdiction/wrong-law-type cases.
- [ ] T069 Include semantic-cache paraphrase pairs.
- [ ] T070 Create a lightweight evaluation runner that reports retrieval/source correctness and cache behavior without adding MLOps infrastructure.

## Phase 11 — Packaging

- [ ] T071 Add Dockerfile for the application.
- [ ] T072 Update Docker Compose to run the complete local PoC where practical.
- [ ] T073 Add `.env.example` with no secrets.
- [ ] T074 Write README with setup, ingestion, indexing, UI startup, architecture, limitations, and evaluation instructions.
- [ ] T075 Add architecture diagram to README.
- [ ] T076 Verify a clean-machine setup from README.
- [ ] T077 Run the complete test suite and fix all blocking failures.
- [ ] T078 Perform a final scope audit ensuring no notebooks or unnecessary MLOps components are required.

## Phase 12 — Final Review

- [ ] T079 Verify every answer containing a legal claim has article-level sources.
- [ ] T080 Verify exact article references take deterministic retrieval into account.
- [ ] T081 Verify repealed provisions are not silently treated as active.
- [ ] T082 Verify cache entries cannot cross jurisdiction/law-type/KB-version boundaries.
- [ ] T083 Verify out-of-scope questions do not trigger unsupported legal answers.
- [ ] T084 Verify Arabic and English equivalent queries retrieve equivalent sources.
- [ ] T085 Verify README commands work in a fresh environment.
