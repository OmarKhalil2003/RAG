# Research Notes and Technical Decisions

## R1 — Why article-level chunks?

Decision: Use legal article as the canonical retrieval unit.

Reason:
- legal article is a natural citation boundary;
- arbitrary fixed windows can split legal provisions;
- citations become directly traceable to article numbers.

The supplied handbook explicitly recommends article-level chunking and splitting only long articles by paragraph while retaining the article number on each child chunk. fileciteturn0file0L56-L61

## R2 — Why structured JSON before embeddings?

Decision: Never embed raw PDF extraction directly.

Reason:
- bilingual PDF columns can interleave;
- article number and hierarchy are required for legal citation and filtering;
- repealed status must be represented explicitly.

The handbook explicitly warns that raw extraction can mix Arabic and English mid-sentence and requires column-aware or positional extraction. fileciteturn0file0L32-L37

## R3 — Why BGE-M3?

Decision: Start with `BAAI/bge-m3`.

Reason:
- multilingual retrieval;
- Arabic + English;
- dense semantic retrieval;
- sparse lexical signal can help with exact legal terminology and identifiers;
- one model family simplifies the PoC.

The model documentation should be consulted at implementation time for the current package/API integration.

## R4 — Why Qdrant?

Decision: Qdrant.

Reason:
- payload metadata filtering;
- dense vectors;
- sparse/hybrid retrieval;
- suitable local deployment;
- clean fit for legal metadata.

## R5 — Why a reranker?

Decision: `BAAI/bge-reranker-v2-m3`.

Reason:
Legal articles often share terminology. Embedding similarity can produce several plausible candidates. A cross-encoder-style reranker provides a second relevance judgment over a small candidate set.

## R6 — Why deterministic article lookup?

Decision: exact article extraction and lookup are mandatory.

Reason:
Article 147 and Article 148 are semantically related. A vector-only system can retrieve the neighboring article. Legal identifiers should be treated as identifiers, not just semantic concepts.

## R7 — Why Redis semantic caching?

Decision: Redis is the semantic response cache.

Reason:
- repeated questions are common;
- semantic paraphrases can reuse responses;
- TTL and metadata isolation are available;
- it is distinct from the legal vector database.

Cache must include knowledge-base and prompt versions.

## R8 — Why Streamlit?

Decision: Streamlit.

Reason:
The requested deliverable is a PoC. A small UI is needed, not a production frontend.

## R9 — Why no LangChain initially?

Decision: no framework dependency unless it materially reduces implementation complexity.

Reason:
The pipeline is easier to inspect when retrieval, caching, and prompting are explicit modules. This also reduces hidden behavior in a legal PoC.

## R10 — Why no MLOps?

Decision: explicitly excluded.

Reason:
The freelance requirement is a RAG capability PoC, not a production MLOps system. The supplied course handbook contains MLOps deliverables such as MLflow, DVC, vLLM/BentoML, RAGAS, and Langfuse, but those are not part of the client's narrower PoC scope. The handbook's RAG checklist identifies those as later project deliverables. fileciteturn0file0L229-L247

## R11 — Evaluation strategy

The minimum evaluation suite should cover:
- exact article lookup;
- Arabic-Indic numerals;
- Arabic semantic queries;
- English semantic queries;
- cross-language retrieval;
- metadata filters;
- repealed articles;
- out-of-scope questions;
- semantic cache hits;
- cache isolation.

The source handbook also recommends checking 20 random extracted articles before trusting the parsed corpus. fileciteturn0file0L40-L42
