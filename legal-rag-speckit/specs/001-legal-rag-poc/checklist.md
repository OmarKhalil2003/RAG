# Legal RAG PoC Review Checklist

## Corpus
- [ ] PDF extraction does not interleave Arabic and English columns.
- [ ] Article numbers are integers.
- [ ] Arabic-Indic numerals are normalized.
- [ ] Arabic source text is non-empty.
- [ ] English source text is preserved where present.
- [ ] Book/chapter/section/topic metadata is populated.
- [ ] Repealed articles are retained and flagged.
- [ ] 20 random extracted articles have been manually inspected.

## Retrieval
- [ ] BGE-M3 is used for multilingual retrieval.
- [ ] Dense retrieval works.
- [ ] Sparse/lexical retrieval works.
- [ ] Metadata filters work.
- [ ] Exact article lookup works.
- [ ] Candidate fusion works.
- [ ] Multilingual reranking works.
- [ ] Final context is limited to relevant candidates.

## Generation
- [ ] LLM cannot answer from unprovided legal knowledge.
- [ ] Unsupported questions produce a grounded refusal.
- [ ] Every legal claim has a source.
- [ ] Article citations are human-readable.
- [ ] Repealed status is visible when relevant.
- [ ] Response language follows the query/user setting.

## Cache
- [ ] Redis semantic cache is enabled.
- [ ] Similar questions can hit the cache.
- [ ] Exact/near-duplicate questions can hit the cache.
- [ ] Jurisdiction is part of cache isolation.
- [ ] Law type is part of cache isolation.
- [ ] Knowledge-base version is part of cache isolation.
- [ ] Prompt version is part of cache isolation.
- [ ] TTL is configured.
- [ ] Cache failure does not weaken legal grounding.

## UI
- [ ] Role selector exists.
- [ ] Jurisdiction selector exists.
- [ ] Law type selector exists.
- [ ] Question box exists.
- [ ] Answer is readable.
- [ ] Sources are visible.
- [ ] Cache status is visible.
- [ ] Internal IDs are not shown as legal citations.

## Repository
- [ ] No notebook is required.
- [ ] No MLOps service is required.
- [ ] `.env` is excluded from Git.
- [ ] README is complete.
- [ ] Tests run successfully.
- [ ] Index can be rebuilt.
- [ ] Docker Compose starts required infrastructure.
