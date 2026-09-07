# Implementation Plan: Arabic Legal RAG PoC

## 1. Technical Context

### Language
Python 3.11+

### UI
Streamlit

### Embeddings
`BAAI/bge-m3`

Rationale:
- multilingual;
- suitable for Arabic and English;
- supports dense and sparse retrieval patterns;
- supports long documents;
- aligns with the requirement to handle semantic meaning while retaining lexical signals.

BGE-M3's model documentation describes multilingual support and hybrid dense/sparse retrieval. The implementation should follow the model's current recommended API rather than hard-coding an obsolete integration.

### Vector Database
Qdrant

Rationale:
- vector similarity search;
- payload metadata filtering;
- dense/sparse hybrid retrieval;
- local Docker execution is straightforward;
- suitable for exact legal metadata fields alongside vectors.

### Reranker
`BAAI/bge-reranker-v2-m3`

Rationale:
- multilingual reranking;
- query/document pair scoring;
- improves precision among semantically similar legal articles.

### Semantic Cache
Redis

Rationale:
- semantic caching support;
- TTL;
- metadata/namespace isolation;
- simple local Docker deployment.

### LLM
Provider-agnostic adapter.

The first implementation should expose an interface such as:

```python
class LLMClient(Protocol):
    def generate(self, *, system_prompt: str, user_prompt: str) -> str: ...
```

The concrete provider is configured through environment variables. The core RAG pipeline must not depend on provider-specific response objects.

### PDF Extraction
PyMuPDF plus a controlled bilingual-layout extraction layer.

The source handbook warns that naive extraction can interleave Arabic and English columns and recommends column-aware/positional handling. fileciteturn0file0L32-L37

### Persistence
- Structured corpus: JSON
- Vector index: Qdrant
- Cache: Redis

No SQL database is required for the PoC.

## 2. Architecture

```text
Streamlit
   |
   v
RAG Orchestrator
   |
   +--> Query Normalizer
   |       |
   |       +--> language
   |       +--> article number
   |       +--> legal scope
   |
   +--> Redis Semantic Cache
   |
   +--> Retriever
   |       |
   |       +--> exact article lookup
   |       +--> dense BGE-M3
   |       +--> sparse retrieval
   |       +--> metadata filters
   |
   +--> Fusion
   |
   +--> BGE Reranker
   |
   +--> Context Builder
   |
   +--> LLM Adapter
   |
   +--> Citation Builder
   |
   +--> Response
```

## 3. Retrieval Strategy

### Step 1 — Normalize Query
- Unicode normalization.
- Arabic whitespace normalization.
- Arabic numeral normalization.
- Lightweight Arabic orthographic normalization.
- Preserve original query.

### Step 2 — Extract Legal Identifiers
Detect patterns such as:
- `Article 147`
- `article 147`
- `المادة 147`
- `المادة ١٤٧`

Return:

```python
QuerySignals(
    article_numbers=[147],
    language="ar",
    jurisdiction="Egypt",
    law_type="civil",
)
```

### Step 3 — Apply Metadata Scope
Construct Qdrant payload filters from:
- jurisdiction;
- law_type;
- document_id;
- optional repeal filter.

### Step 4 — Exact Retrieval
If an article number is detected:
- retrieve matching article records directly;
- add them to the candidate pool;
- do not discard them because their dense similarity is lower.

### Step 5 — Dense Retrieval
Embed the normalized query with BGE-M3 and retrieve top-K dense candidates.

### Step 6 — Sparse Retrieval
Retrieve lexical matches for legal terminology, article references, and identifiers.

### Step 7 — Fusion
Use Reciprocal Rank Fusion or an equivalent deterministic fusion method.

Candidate pool target: 10–20 documents.

### Step 8 — Reranking
Apply `bge-reranker-v2-m3` to candidate query/document pairs.

Final context target: 3–5 articles/chunks.

### Step 9 — Context Validation
Before generation:
- remove candidates outside metadata scope;
- preserve article number;
- preserve repeal status;
- deduplicate same article;
- enforce minimum relevance threshold.

### Step 10 — Generation
Construct a role-specific prompt around the same legal context.

## 4. Article Chunking

Canonical chunk:
- one article;
- article number retained;
- hierarchy retained;
- Arabic and English source text retained.

For exceptionally long articles, split by numbered paragraph while duplicating article metadata onto every child chunk.

The source handbook explicitly recommends article-level chunking rather than arbitrary token windows. fileciteturn0file0L56-L61

## 5. Semantic Cache Design

Cache record:

```text
cache_namespace
query_embedding
normalized_query
answer
sources
jurisdiction
law_type
document_id
knowledge_base_version
prompt_version
created_at
expires_at
```

Cache lookup scope:
```text
jurisdiction
AND law_type
AND document_id
AND knowledge_base_version
AND prompt_version
```

Cache hit response:
```json
{
  "answer": "...",
  "sources": [...],
  "cached": true
}
```

Cache miss response:
```json
{
  "answer": "...",
  "sources": [...],
  "cached": false
}
```

The cache MUST NOT be allowed to return an answer from a different legal scope.

## 6. Role Prompting

### Lawyer
- precise;
- source-first;
- minimal simplification;
- explicit article references.

### Citizen
- plain language;
- explain legal terminology;
- preserve source citations.

### Law Student
- explain the rule;
- identify the relevant article;
- optionally summarize the legal concept.

### Researcher
- detailed;
- emphasize related retrieved provisions;
- preserve source hierarchy.

Role changes wording only. It does not expand the legal knowledge available to the system.

## 7. API/Application Contract

Primary internal function:

```python
answer = rag_service.ask(
    question: str,
    role: UserRole,
    jurisdiction: str,
    law_type: str,
)
```

Response:

```python
RAGResponse(
    answer: str,
    sources: list[Source],
    cached: bool,
    query_signals: QuerySignals,
)
```

The PoC may expose a FastAPI endpoint later, but Streamlit can directly call the application service initially.

## 8. Data Flow

```text
PDF
 -> parser
 -> bilingual article extraction
 -> normalization
 -> validation
 -> JSON
 -> embedding/indexing
 -> Qdrant

User query
 -> normalize
 -> detect article number
 -> Redis semantic cache
 -> exact + dense + sparse retrieval
 -> metadata filtering
 -> RRF
 -> reranker
 -> context builder
 -> LLM
 -> citations
 -> cache write
 -> UI
```

## 9. Project Structure

```text
legal-rag/
├── app/
│   └── streamlit_app.py
├── src/
│   └── legal_rag/
│       ├── __init__.py
│       ├── config.py
│       ├── models.py
│       ├── pipeline.py
│       ├── ingestion/
│       │   ├── parser.py
│       │   ├── normalizer.py
│       │   ├── validator.py
│       │   └── schema.py
│       ├── retrieval/
│       │   ├── query_parser.py
│       │   ├── qdrant.py
│       │   ├── hybrid.py
│       │   └── reranker.py
│       ├── cache/
│       │   └── semantic_cache.py
│       └── generation/
│           ├── llm.py
│           └── prompts.py
├── scripts/
│   ├── ingest.py
│   └── build_index.py
├── data/
│   ├── raw/
│   └── processed/
├── evaluation/
│   └── questions.json
├── tests/
├── specs/
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── README.md
```

## 10. Error Handling

- Invalid/empty question → validation error.
- No retrieval candidates → grounded refusal.
- Article not found → state that the referenced article is not in the corpus.
- Unsupported law type → scope error.
- Unsupported jurisdiction → scope error.
- LLM failure → application error without fabricated answer.
- Redis failure → continue without cache if retrieval/generation remains valid.
- Qdrant failure → fail clearly; do not answer from LLM prior knowledge.
- Malformed corpus record → fail indexing and report record ID.

## 11. Constitution Check

PASS:
- Legal grounding is enforced by context-only generation.
- Citations are explicit.
- Article numbers have deterministic handling.
- Repealed records are preserved.
- Metadata filters are first-class.
- Bilingual source text is preserved.
- Cache scope includes legal context.
- No MLOps infrastructure is introduced.
- Components are separated.

## 12. Implementation Phases

### Phase 0 — Corpus
Extract, normalize, validate, and serialize the Civil Code.

### Phase 1 — Index
Create Qdrant collection and index article records.

### Phase 2 — Retrieval
Implement query parsing, exact lookup, dense/sparse retrieval, fusion, and reranking.

### Phase 3 — Generation
Implement grounded prompts, role behavior, and citations.

### Phase 4 — Cache
Implement Redis semantic cache with scope isolation.

### Phase 5 — UI
Implement Streamlit.

### Phase 6 — Evaluation
Run deterministic retrieval/grounding test suite.

### Phase 7 — Packaging
Docker Compose, README, environment configuration, and final cleanup.
