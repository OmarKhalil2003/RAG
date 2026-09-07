# Arabic Legal RAG PoC Constitution

## Core Principles

### I. Legal Grounding Is Mandatory
The system MUST answer legal questions only from retrieved source documents in the configured knowledge base. The generation layer MUST NOT present unsupported legal claims, invented article numbers, invented provisions, or fabricated citations as facts.

If the retrieved context is insufficient, the system MUST explicitly state that the available knowledge base does not contain enough information to answer the question.

### II. Citation Is Part of the Answer
Every substantive legal answer MUST expose verifiable legal sources. Sources MUST identify the law and article number. Internal chunk IDs, vector IDs, or opaque database identifiers MUST NOT be presented as the primary legal citation.

### III. Article-Level Data Integrity
Legal documents MUST be transformed into structured records before indexing. The canonical retrieval unit SHOULD be a legal article. Article numbers MUST be normalized into machine-sortable integers while preserving the original legal text.

Repealed provisions MUST be retained and explicitly marked rather than silently deleted.

### IV. Deterministic Handling of Legal Identifiers
Exact legal identifiers such as article numbers MUST NOT depend exclusively on semantic similarity. When a query contains an article reference, the system MUST normalize and extract it and use deterministic metadata/exact lookup in addition to semantic retrieval.

### V. Retrieval Must Respect Legal Scope
Retrieval MUST support metadata constraints including at least jurisdiction, law type, document/law identifier, and repeal status. The system MUST NOT mix jurisdictions or law families merely because their language is semantically similar.

### VI. Bilingual Fidelity
The PoC MUST support Arabic and English questions against the same bilingual legal corpus. Arabic normalization MUST be applied consistently without destroying the source text. The system MUST preserve Arabic and English source text separately for answer generation and citation.

### VII. Cache Safety
Semantic caching MUST NOT bypass legal-scope isolation. Cache keys or cache metadata MUST include the relevant knowledge-base context, including jurisdiction, law type, knowledge-base version, and prompt version. A cache hit MUST be treated as a performance optimization, not as a separate source of legal truth.

### VIII. Simplicity Over Infrastructure
This is a freelance proof of concept. The implementation MUST remain a focused RAG application. MLOps platforms, experiment tracking, DVC, distributed serving, monitoring stacks, and notebooks are out of scope unless explicitly requested later.

### IX. Test the Failure Modes
Tests MUST include exact article queries, Arabic-Indic numerals, Arabic and English paraphrases, metadata filtering, repealed articles, and out-of-scope questions. A system that answers fluently but retrieves the wrong law is considered incorrect.

### X. Separation of Concerns
Parsing/normalization, indexing, retrieval, reranking, caching, generation, and UI concerns MUST be separated into modules. The UI MUST NOT contain retrieval or prompt-construction logic.

## Non-Goals

The initial PoC does not provide:
- personalized legal representation;
- a guarantee that an answer reflects current law outside the indexed corpus;
- court filing or legal-document generation workflows;
- legal case prediction;
- legal advice based on a user's personal facts;
- authentication/authorization infrastructure;
- multi-tenant production deployment;
- MLOps/observability infrastructure.

## Engineering Standards

- Python project layout with `pyproject.toml`.
- Type hints for application code.
- Configuration through environment variables with `.env.example`.
- No notebook-based implementation.
- Unit tests for normalization, query parsing, retrieval filters, caching isolation, and citation construction.
- README must allow a reviewer to install and run the PoC without asking the author for undocumented steps.

## Governance

This constitution is the highest-level project constraint. The feature specification defines WHAT the system must do; the implementation plan defines HOW it will be built. If implementation convenience conflicts with legal grounding, citation integrity, or deterministic legal-identifier handling, the latter wins.

Version: 1.0.0
Status: Ratified for the initial freelance PoC
