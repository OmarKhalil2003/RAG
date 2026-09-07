# Data Model

## 1. LegalArticle

```json
{
  "chunk_id": "civil_147",
  "document_id": "egyptian_civil_code_1948",
  "jurisdiction": "Egypt",
  "law_type": "civil",
  "law_name_ar": "القانون المدني المصري",
  "law_name_en": "Egyptian Civil Code",
  "law_year": 1948,
  "article_number": 147,
  "book": "Obligations or Personal Rights",
  "chapter": "Sources of Obligations",
  "section": "Contracts",
  "topic": "The Effects of a Contract",
  "text_ar": "...",
  "text_en": "...",
  "is_repealed": false,
  "source_page": 34,
  "citation": "Egyptian Civil Code, Article 147"
}
```

## 2. IndexedChunk

```json
{
  "chunk_id": "civil_147",
  "dense_vector": "<Qdrant vector>",
  "sparse_vector": "<Qdrant sparse vector>",
  "payload": {
    "document_id": "egyptian_civil_code_1948",
    "jurisdiction": "Egypt",
    "law_type": "civil",
    "article_number": 147,
    "book": "...",
    "chapter": "...",
    "section": "...",
    "topic": "...",
    "is_repealed": false,
    "citation": "Egyptian Civil Code, Article 147"
  }
}
```

## 3. QuerySignals

```python
class QuerySignals:
    original_query: str
    normalized_query: str
    language: Literal["ar", "en", "unknown"]
    article_numbers: list[int]
    jurisdiction: str | None
    law_type: str | None
```

## 4. RetrievalResult

```python
class RetrievalResult:
    chunk_id: str
    article_number: int
    score: float
    retrieval_sources: list[str]
    metadata: dict
    text_ar: str
    text_en: str
```

## 5. SourceCitation

```python
class SourceCitation:
    document_id: str
    law_name: str
    article_number: int
    citation: str
    is_repealed: bool
```

## 6. RAGResponse

```python
class RAGResponse:
    answer: str
    sources: list[SourceCitation]
    cached: bool
    retrieval_count: int
```

## 7. CacheEntry

```json
{
  "cache_key_namespace": "legal-rag",
  "normalized_query": "...",
  "answer": "...",
  "sources": [],
  "jurisdiction": "Egypt",
  "law_type": "civil",
  "document_id": "egyptian_civil_code_1948",
  "knowledge_base_version": "v1",
  "prompt_version": "v1",
  "created_at": "...",
  "expires_at": "..."
}
```

## Metadata rules

Required:
- `jurisdiction`
- `law_type`
- `document_id`
- `article_number`
- `is_repealed`

Recommended:
- `book`
- `chapter`
- `section`
- `topic`
- `law_year`

## Identifier rules

`article_number` is an integer.

Arabic-Indic digits are normalized before persistence:
- `٠١٢٣٤٥٦٧٨٩` → `0123456789`

Source text is never overwritten by normalized text.
