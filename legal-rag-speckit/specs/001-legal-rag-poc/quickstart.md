# Quickstart

## Prerequisites

- Python 3.11+
- Docker
- Docker Compose
- API key for the selected LLM provider

## 1. Install

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -e .
```

## 2. Configure

Copy:

```bash
cp .env.example .env
```

Configure:

```text
LLM_PROVIDER=...
LLM_API_KEY=...
LLM_MODEL=...
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379/0
KNOWLEDGE_BASE_VERSION=v1
PROMPT_VERSION=v1
```

## 3. Start infrastructure

```bash
docker compose up -d qdrant redis
```

## 4. Build the corpus

```bash
python scripts/ingest.py
```

Expected output:

```text
Parsed articles: N
Validation: PASS
Output: data/processed/egyptian_civil_code.json
```

## 5. Build the vector index

```bash
python scripts/build_index.py
```

Expected output:

```text
Collection: legal_articles
Indexed: N
Status: READY
```

## 6. Start UI

```bash
streamlit run app/streamlit_app.py
```

## 7. Run tests

```bash
pytest
```

## Expected UI behavior

The UI should expose:
- user role;
- jurisdiction;
- law type;
- question input;
- answer;
- article sources;
- cache status.

## Example questions

Arabic:

```text
ما هي آثار العقد بالنسبة للمتعاقدين؟
```

Exact article:

```text
ماذا تنص المادة ١٤٧؟
```

English:

```text
What does Article 147 of the Egyptian Civil Code provide?
```

Out of scope:

```text
What is the penalty for murder in Egypt?
```

The final example should not result in an invented criminal-law answer because the initial corpus is the Egyptian Civil Code.
