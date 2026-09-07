# Feature Specification: Arabic Legal RAG PoC

**Feature:** 001-legal-rag-poc  
**Status:** Draft for implementation  
**Scope:** Freelance proof of concept  
**Primary corpus:** Egyptian Civil Code, bilingual Arabic/English PDF

## 1. Problem Statement

Legal users need to ask questions over legal documents in Arabic or English and receive answers grounded in the available legal corpus.

The PoC must demonstrate that a RAG system can:
1. ingest a bilingual legal document into structured article-level records;
2. retrieve the correct legal provisions using semantic, lexical, metadata, and exact-identifier signals;
3. answer in the user's language;
4. cite the relevant law and article number;
5. distinguish repealed provisions;
6. avoid answering from unsupported knowledge;
7. reduce repeated-question latency through semantic caching.

The uploaded project handbook establishes the same core constraints: structured JSON per article, article-level chunking, Arabic/English handling, normalized article numbers, repealed-article flags, metadata hierarchy, and article-number citations. fileciteturn0file0L18-L30 fileciteturn0file0L40-L67

## 2. Users

### US1 — Lawyer
A lawyer wants precise legal retrieval with explicit article citations.

### US2 — Regular Citizen
A citizen wants a plain-language explanation of a legal provision without being presented with unsupported legal claims.

### US3 — Law Student
A student wants a concise explanation supported by the relevant legal articles.

### US4 — Legal Researcher
A researcher wants broader retrieval and source visibility for a legal topic.

The initial implementation uses one RAG pipeline. User role changes response style and presentation, not the underlying legal source of truth.

## 3. In-Scope Knowledge Base

Initial corpus:
- Jurisdiction: Egypt
- Law type: civil
- Law: Egyptian Civil Code
- Source: provided bilingual Arabic/English PDF
- Granularity: article

The corpus must be designed so additional law types can be added later, including criminal, commercial, labor, family, or other jurisdictions, without changing the core retrieval interface.

## 4. User Stories and Acceptance Scenarios

### US1 — Ask a Legal Question

**As a lawyer, I want to ask a legal question so that I can retrieve relevant legal provisions with citations.**

Acceptance:
- Given a question covered by the corpus, when the user submits it, then the answer is generated from retrieved legal context.
- The response includes at least one source when the answer makes a legal claim.
- Each source identifies the law and article number.
- The answer does not expose internal vector/chunk identifiers as the legal citation.

### US2 — Arabic Question

**As an Arabic-speaking user, I want to ask in Arabic and receive an Arabic answer.**

Acceptance:
- Arabic questions are accepted.
- Arabic-Indic article numerals such as `١٤٧` are normalized for retrieval.
- Arabic source text is preserved.
- The answer is returned in Arabic unless the user explicitly selects another response language.

### US3 — English Question

**As an English-speaking user, I want to ask in English against the same corpus.**

Acceptance:
- English questions retrieve the corresponding bilingual legal article.
- Equivalent Arabic and English questions should retrieve the same article for the evaluation cases.

The handbook explicitly identifies bilingual retrieval as a useful evaluation of whether the embedding model handles Arabic and English consistently. fileciteturn0file0L68-L73

### US4 — Exact Article Lookup

**As a legal user, I want a question containing an article number to retrieve that article reliably.**

Acceptance:
- `Article 147`, `المادة 147`, and `المادة ١٤٧` resolve to the same normalized article number.
- Exact article lookup participates in retrieval.
- Semantic similarity alone is not the only mechanism used for article-number queries.

### US5 — Law-Type Filtering

**As a user, I want retrieval restricted to the selected law type.**

Acceptance:
- The UI exposes the currently indexed law type.
- Retrieval can filter by `law_type`.
- Adding another law type later does not require redesigning the retrieval API.

### US6 — Jurisdiction Filtering

**As a user, I want retrieval restricted to the selected jurisdiction.**

Acceptance:
- Retrieval supports `jurisdiction` as metadata.
- A future Saudi document cannot be returned for an Egypt-only request unless explicitly selected.

### US7 — Repealed Provision

**As a user, I want the system to recognize when an article is marked repealed.**

Acceptance:
- Repealed articles remain in the corpus.
- Retrieval metadata exposes `is_repealed`.
- The answer explicitly distinguishes historical/repealed provisions from active provisions.

The handbook states that repealed articles should be flagged rather than deleted because "that article no longer exists" can be the correct answer. fileciteturn0file0L43-L47

### US8 — Out-of-Scope Question

**As a user, I want the system to refuse unsupported questions rather than hallucinate.**

Acceptance:
- A question about a law not present in the knowledge base does not receive an invented legal answer.
- The response states that the available corpus does not contain sufficient information.
- No unsupported article citation is generated.

### US9 — Semantic Cache

**As a user, I want repeated or near-equivalent questions to return faster.**

Acceptance:
- The first eligible query can populate Redis.
- A sufficiently similar repeat/paraphrase can hit the semantic cache.
- Cache entries are isolated by jurisdiction, law type, knowledge-base version, and prompt version.
- The UI/API indicates whether the response was served from cache.

### US10 — Role-Aware Answering

**As a user, I want the response style to match my role.**

Acceptance:
- Lawyer: precise, source-forward wording.
- Citizen: simpler explanation.
- Student: explanatory/educational wording.
- Researcher: detailed source-oriented wording.
- The role MUST NOT cause the system to invent additional legal sources.

## 5. Functional Requirements

### Corpus and ingestion

**FR-001** The system SHALL parse the supplied bilingual legal PDF into structured JSON records.

**FR-002** Each article record SHALL contain at least:
`document_id`, `jurisdiction`, `law_type`, `law_name_ar`, `law_name_en`, `law_year`, `article_number`, `book`, `chapter`, `section`, `topic`, `text_ar`, `text_en`, `is_repealed`, `source_page`, and `citation`.

**FR-003** The ingestion pipeline SHALL normalize Arabic-Indic article numbers to Western integer values.

**FR-004** The ingestion pipeline SHALL preserve the original Arabic and English text separately.

**FR-005** The ingestion pipeline SHALL normalize Arabic orthographic variants consistently for retrieval while retaining source text unchanged for citation/generation.

**FR-006** The ingestion pipeline SHALL flag repealed articles instead of deleting them.

**FR-007** The system SHALL validate that each article has a valid integer article number, non-empty Arabic text, and a sane record length.

The handbook recommends these validation checks before embedding. fileciteturn0file0L50-L55

### Retrieval

**FR-008** The system SHALL use multilingual embeddings suitable for Arabic and English.

**FR-009** The system SHALL support dense semantic retrieval.

**FR-010** The system SHALL support lexical/sparse retrieval for exact legal terms and identifiers.

**FR-011** The system SHALL support metadata filtering by at least `jurisdiction`, `law_type`, `document_id`, and `is_repealed`.

**FR-012** The system SHALL detect explicit article-number references before retrieval.

**FR-013** Article-number retrieval SHALL use deterministic metadata/exact matching in addition to semantic retrieval.

**FR-014** The system SHALL fuse dense, sparse, and exact-identifier candidates before reranking.

**FR-015** The system SHALL rerank the candidate set before generation.

### Generation

**FR-016** The generator SHALL answer only from retrieved legal context.

**FR-017** The generator SHALL cite legal claims using law/article citations.

**FR-018** The generator SHALL answer in the user's language.

**FR-019** The generator SHALL explicitly state when retrieved context is insufficient.

**FR-020** The generator SHALL distinguish repealed provisions when the answer relies on them.

**FR-021** The system SHALL not present the PoC as a substitute for legal counsel or current official legal verification.

### Caching

**FR-022** The system SHALL implement Redis-based semantic response caching.

**FR-023** Cache eligibility SHALL include the legal retrieval scope and knowledge-base version.

**FR-024** A cache hit SHALL return the previously generated answer and its sources.

**FR-025** Cache invalidation SHALL be possible through TTL and knowledge-base/prompt version changes.

### UI

**FR-026** The system SHALL provide a Streamlit UI.

**FR-027** The UI SHALL allow selection of user role.

**FR-028** The UI SHALL expose jurisdiction and law-type scope for the indexed corpus.

**FR-029** The UI SHALL show answer, sources, and cache status.

### Engineering

**FR-030** The repository SHALL contain no notebooks required to run the PoC.

**FR-031** The repository SHALL provide a reproducible ingestion/indexing command.

**FR-032** The repository SHALL provide automated tests for critical retrieval and grounding behavior.

**FR-033** MLOps systems such as MLflow, DVC, Langfuse, Prometheus, Grafana, and production serving platforms SHALL remain out of scope.

## 6. Non-Functional Requirements

**NFR-001 — Grounding:** Unsupported legal claims must be rejected or clearly qualified.

**NFR-002 — Retrieval correctness:** Exact article queries must prioritize the referenced article.

**NFR-003 — Bilingual behavior:** Equivalent Arabic/English queries should retrieve equivalent legal sources.

**NFR-004 — Isolation:** Retrieval must not cross the selected jurisdiction/law scope.

**NFR-005 — Explainability:** Every generated legal answer must expose source articles.

**NFR-006 — Maintainability:** RAG stages must be independently testable.

**NFR-007 — PoC simplicity:** The system should be runnable locally with a small number of commands.

## 7. Edge Cases

1. Arabic-Indic article number: `١٤٧`.
2. Western article number embedded in Arabic text.
3. Question references an article that is absent.
4. Question references a repealed article.
5. User asks about a law type not indexed.
6. User asks about another jurisdiction.
7. Arabic question uses diacritics.
8. Arabic question uses alternate alef/hamza forms.
9. Very short query such as `المادة 147`.
10. Question semantically resembles a neighboring article.
11. Exact article number conflicts with semantic ranking.
12. Cache contains a similar question from a different law type.
13. Cache contains an answer generated under an older prompt version.
14. Retrieved documents are semantically relevant but belong to the wrong jurisdiction.
15. No retrieved candidate passes the configured relevance threshold.

## 8. Success Criteria

**SC-001:** A reviewer can ingest the supplied corpus and launch the UI using documented commands.

**SC-002:** Exact queries for selected article numbers retrieve the correct article in Arabic and English.

**SC-003:** Arabic and English paraphrases of the same legal concept retrieve the same relevant article set in the evaluation suite.

**SC-004:** Metadata filters prevent retrieval outside the selected jurisdiction/law type.

**SC-005:** Repealed article tests return the repealed status rather than silently treating the article as active.

**SC-006:** Out-of-scope tests do not produce unsupported legal citations.

**SC-007:** Repeated/paraphrased eligible queries can demonstrate a Redis semantic-cache hit.

**SC-008:** The UI displays the generated answer and article-level sources.

**SC-009:** The repository can be understood and run without a notebook.

## 9. Explicit Out of Scope

- MLflow
- DVC
- RAGAS as a required dependency
- Langfuse
- Prometheus/Grafana
- BentoML
- vLLM as a required serving layer
- canary deployment
- production authentication
- user account management
- case management
- legal advice workflows
- automated legal opinion generation
- web search as a fallback source
- automatic ingestion of arbitrary internet laws
