# Bilingual Arabic Legal RAG Assistant

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 32 Passed](https://img.shields.io/badge/tests-32%20passed-brightgreen.svg)]()
[![Evaluation: 25/25 Passed](https://img.shields.io/badge/eval-25%2F25%20passed%20(100%25)-brightgreen.svg)]()
[![RAGAS: 93.7% Faithfulness | 98.1% Precision](https://img.shields.io/badge/RAGAS-93.7%25%20Faithfulness%20%7C%2098.1%25%20Precision-blue.svg)]()
[![Retrieval: Hybrid RRF](https://img.shields.io/badge/retrieval-BGE--M3%20%2B%20BM25%20%2B%20Exact-orange.svg)]()
[![LLMs: Gemini 3.1 & OpenAI](https://img.shields.io/badge/LLMs-Gemini%203.1%20%7C%20OpenAI%20%7C%20Mock-purple.svg)]()

**Proof of Concept (PoC)** for a **Multi-Statute Bilingual Arabic Legal RAG Assistant** indexing Egyptian legislation at article granularity:
1. **Egyptian Civil Code (القانون المدني المصري)** — Law No. 131 of 1948 (1,149 articles, 56 repealed provisions flagged).
2. **Egyptian Arbitration Law (قانون التحكيم المصري)** — Law No. 27 of 1994 (58 articles on domestic and international arbitration).

Engineered for precision legal research, deterministic article-number lookup, hybrid dense + sparse retrieval, cross-encoder reranking, hard grounding gates, role-aware legal advisory, Arabic OCR ligature repair, cross-statute isolation, and isolated semantic caching.

---

## Architecture

```text
                    ┌──────────────────────────────┐
                    │    Streamlit Web UI          │
                    │ (Statute Dropdown / Role)    │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │   Query Processing Engine    │
                    │ • Language detection (ar/en) │
                    │ • Deterministic Article #    │
                    │   (١٤٧ -> 147 normalization) │
                    │ • Conservative Normalization │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │   Redis Semantic Cache       │
                    │ (namespaced by statute & role│
                    └──────────┬────────┬──────────┘
                             HIT        MISS
                               │          │
                               │          ▼
                               │   ┌────────────────────────┐
                               │   │ Exact Article Lookup   │
                               │   │ (Qdrant payload filter)│
                               │   └──────────┬─────────────┘
                               │              │
                               │              ▼
                               │   ┌────────────────────────┐
                               │   │ Dense BGE-M3 (Top 20)  │
                               │   │ BM25 Sparse (Top 20)   │
                               │   └──────────┬─────────────┘
                               │              │
                               │              ▼
                               │   ┌────────────────────────┐
                               │   │ RRF Fusion + Exact     │
                               │   │ Priority Injection     │
                               │   └──────────┬─────────────┘
                               │              │
                               │              ▼
                               │   ┌────────────────────────┐
                               │   │ BGE Reranker v2-m3     │
                               │   │ (Top 5 Candidates)     │
                               │   └──────────┬─────────────┘
                               │              │
                               │              ▼
                               │   ┌────────────────────────┐
                               │   │ Hard Grounding Gate    │
                               │   │ (Floor & Scope Check)  │
                               │   └──────────┬─────────────┘
                               │              │
                               │              ▼
                               │   ┌────────────────────────┐
                               │   │ Role-Aware Generator   │
                               │   │ (Gemini 3.1 / OpenAI / │
                               │   │  Deterministic Mock)   │
                               │   └──────────┬─────────────┘
                               │              │
                               │              ▼
                               │   ┌────────────────────────┐
                               │   │ Deterministic          │
                               │   │ Citation Builder       │
                               │   └──────────┬─────────────┘
                               │              │
                               └───────┬──────┘
                                       ▼
                       Answer + Citations + Cache Badge
```

---

## Architectural Decisions

### 1. Multi-Statute Registry & Cross-Statute Isolation
Statutes are registered in `data/corpora_registry.json`. Queries are partitioned by `jurisdiction` and `law_type`:
- **Egyptian Civil Code**: `law_type="civil"`, 1,149 articles.
- **Egyptian Arbitration Law**: `law_type="arbitration"`, 58 articles.
Deterministic UUID point IDs (`uuid5(document_id, article_number)`) and namespaced Redis cache keys guarantee that Article 1 of the Arbitration Law never collides with Article 1 of the Civil Code.

### 2. Deterministic Legal Identifier Handling
Embeddings alone cannot differentiate neighboring articles (e.g., Article 147 vs Article 148). When a query mentions an article number in Western (`Article 147`, `147`) or Arabic-Indic numerals (`المادة ١٤٧`), the system:
1. Normalizes digits: `١٤٧` → `147`.
2. Executes a deterministic Qdrant payload match (`article_number == 147 AND law_type == active_law`).
3. Injects the exact article into the candidate pool with top priority alongside dense semantic matches.

### 3. Hybrid Dense (BGE-M3) + Sparse (BM25) Retrieval + Cross-Encoder Reranker
- **BGE-M3 Dense Search**: Captures bilingual conceptual meaning, contractual principles, and legal remedies (1024-dim).
- **BM25 Sparse Search**: Matches exact statutory terminology, definitions, and code phrasing over 1,207 total indexed articles.
- **Reciprocal Rank Fusion (RRF)**: Merges dense and sparse rankings ($k=60$).
- **BAAI/bge-reranker-v2-m3**: Cross-encoder scoring to filter out peripheral candidates and deliver top-5 authoritative provisions.

### 4. Hard Grounding Gate
Before invoking the LLM, a hard validation gate ensures:
- Queries with zero candidates or scores below the relevance floor trigger an immediate grounded refusal.
- Explicitly queried articles absent from the active statute range (e.g., Article 9999, or Article 100 in Arbitration Law) trigger grounded refusal.
- Domain mismatches (e.g., criminal queries against civil/arbitration statutes) are refused at the gate.
- The LLM receives **only** validated post-reranking candidates.

### 5. Arabic OCR & Ligature Repair Engine
In Arabic legal PDFs, extraction frequently introduces reverse-ligature artifacts (e.g., `لا` extracted as `ال`) and split Hamza forms. The system includes an automated legal typography cleaner (`clean_arabic_ocr_artifacts`):
- **Negative Particles**: Inverted Lam-Alef before verbs (`ال ينصرف` → `لا ينصرف`, `ال يجوز` → `لا يجوز`, `بحيث ال` → `بحيث لا`).
- **Alif-Hamza Ligatures**: Corrects reversed Alif-Lam-Hamza (`األثر` → `الأثر`, `األصيل` → `الأصيل`, `األول` → `الأول`, `اآلخر` → `الآخر`).
- **Tanween Normalization**: Resolves broken tanween forms (`باطال` → `باطلاً`, `قابال` → `قابلاً`, `مستحيال` → `مستحيلاً`).
- **Legal Phrasing**: Heuristic repairs for OCR defects (`وسد الموازنة` → `وبعد الموازنة`, `الهالتين` → `الحالتين`, `ما لم يرف به` → `ما لم يوف به`).
- **Clause Numbering**: Normalizes RTL/LTR bracket anomalies (`( ١ (` → `(١)`).

### 6. Multi-Role Grounded Personas
Role alters **explanation style, not legal evidence**:
- **👨‍⚖️ Lawyer (محامٍ)**: Formal statutory memorandum, direct statutory basis, elements, and remedies.
- **👤 Regular Citizen (مواطن)**: Plain-language explanation of rights and liabilities with direct citations.
- **🎓 Law Student (طالب حقوق)**: Doctrinal analysis, rule rationale, legal principles, and statutory construction.
- **🔬 Legal Researcher (باحث قانوني)**: Systematic statutory exegesis cross-referencing code hierarchy (Book, Chapter, Section).

### 7. Multi-Provider LLM Intelligence (Gemini 3.1 / 3, OpenAI, Offline Mock)
- **Google Gemini**: Default support for high-throughput, generous-quota models (`gemini-3.1-flash-lite`, `gemini-3-flash-preview`, `gemini-flash-latest`, `gemini-3.5-flash`). Automatic cascade fallback ensures uninterrupted responses if a model reaches a quota limit.
- **OpenAI**: Support for `gpt-4o-mini` and `gpt-4o`.
- **Deterministic Mock Generator**: Offline evaluation mode with zero external API calls or token costs.

### 8. Isolated Redis Semantic Cache
- Isolated namespace: `legal_cache:{jurisdiction}:{law_type}:{user_role}:{kb_version}:{prompt_version}:{hash}`.
- Evaluated via cosine distance: `SEMANTIC_CACHE_DISTANCE_THRESHOLD=0.10`.
- Cross-statute cache isolation: query for Article 1 in Civil Code will never hit cached Article 1 of Arbitration Law.
- Cache flush available via Streamlit sidebar button or CLI script (`scripts/clear_cache.py`).

### 9. RAGAS Multi-Statute Assessment Framework
Integrated **RAGAS (Retrieval Augmented Generation Assessment)** evaluation pipeline across both Egyptian statutory corpora:
- **Zero-Cost Local Embeddings**: `LocalBGEM3Embeddings` provides dense LangChain-compatible embeddings using the local BGE-M3 model, avoiding external embedding API costs.
- **Dual-Mode LLM Judges**: Supports Gemini (`gemini-3.1-flash-lite`), OpenAI (`gpt-4o-mini`), or an offline deterministic evaluator (`DeterministicLegalEvaluator`).
- **Four Core RAGAS Metrics**:
  - **Faithfulness**: Verifies that every assertion in the legal advisory is firmly grounded in retrieved statutory text (preventing legal hallucinations).
  - **Answer Relevancy**: Assesses how directly and pertinently the response addresses the user's specific statutory query.
  - **Context Precision**: Measures whether the authoritative statutory articles are prioritized at top ranks in the retrieval results.
  - **Context Recall**: Verifies that all mandatory statutory provisions needed to answer the inquiry were retrieved.

---

## Installation & Setup

### Prerequisites
* Python 3.10 or higher
* Git

### Step 1: Install Dependencies

You can install all project dependencies using either **`pyproject.toml`** or **`requirements.txt`**:

```bash
# Option A: Install via pyproject.toml
pip install -e .

# Option B: Install via requirements.txt
pip install -r requirements.txt
```

> **Note for Windows Users**: If running in a virtual environment, activate it first:
> ```powershell
> python -m venv .venv
> .venv\Scripts\Activate.ps1
> pip install -e .
> ```

### Step 2: Configure Environment (`.env`)

Copy the example configuration:
```bash
cp .env.example .env
```

Edit `.env` with your preferred settings:
```ini
# LLM Provider: 'gemini', 'openai', or 'mock'
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.1-flash-lite

# Optional: OpenAI Configuration
OPENAI_API_KEY=your_openai_key_here
OPENAI_MODEL=gpt-4o-mini

# Qdrant Vector Store (Leave blank to use embedded disk mode without Docker)
QDRANT_URL=
QDRANT_STORAGE_PATH=data/qdrant_db

# Redis Semantic Cache (Falls back to in-memory cache if Redis is offline)
REDIS_URL=redis://localhost:6379/0
SEMANTIC_CACHE_DISTANCE_THRESHOLD=0.10
SEMANTIC_CACHE_TTL=3600

# Local Development Mode
LOCAL_DEV_MODE=true
```

---

## Quickstart in 3 Commands

### 1. Ingest Corpora & Validate
Parses all bilingual statute PDFs in `data/law_books/` registered in `data/corpora_registry.json`:
```bash
python scripts/ingest.py
```
*Output:*
- *Egyptian Civil Code: `1,149 articles extracted, 56 repealed provisions flagged, 0 missing.`*
- *Egyptian Arbitration Law: `58 articles extracted, 0 repealed, 0 missing.`*

### 2. Build Hybrid Vector & Lexical Index
Generates BGE-M3 dense embeddings, unified BM25 sparse index (1,207 articles), and populates Qdrant:
```bash
python scripts/build_index.py
```

### 3. Launch Streamlit Web UI
> 💡 **Important for Windows PowerShell**: Run using `python -m streamlit run` to avoid the PowerShell path error (`The term 'streamlit' is not recognized`):

```bash
python -m streamlit run app/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser. Use the **Select Active Statute** dropdown in the sidebar to seamlessly switch between the **Egyptian Civil Code** and **Egyptian Arbitration Law**.

---

## Evaluation & Ablation Benchmarks

### Run Automated Evaluation Suite
Evaluates Recall@1/3/5, MRR, Exact Article Accuracy, Citation Integrity, Grounded Refusal, and Semantic Cache hits across 25 curated test cases:
```bash
python evaluation/run_eval.py
```

**Benchmark Results**:
```text
Total Test Cases:            25
Exact Article Accuracy:      10/10 (100.0%)
Recall@1:                    21/22 (95.5%)
Recall@3:                    22/22 (100.0%)
Recall@5:                    22/22 (100.0%)
Mean Reciprocal Rank (MRR):  0.9697
Grounded Refusal Correct:    3/3 (100.0%)
Citation Integrity Rate:     22/22 (100.0%)
Scope Isolation Rate:        22/22 (100.0%)
Semantic Cache Paraphrase:   Passed
```

### Run Retrieval Ablation Benchmark
Measures the incremental value of each architectural layer:
```bash
python evaluation/run_ablation.py
```
*Demonstrates progression from Dense-Only (81.8% Recall@1) to Full Hybrid + Exact + Reranker (95.5% Recall@1, 100% Recall@3).*

### Run RAGAS Assessment Suite
Evaluates Faithfulness, Answer Relevancy, Context Precision, and Context Recall across both statutes using the 35-case benchmark:

```bash
# Run complete multi-statute benchmark (all 35 cases, deterministic offline mode)
python evaluation/run_ragas_eval.py --offline

# Run with an LLM judge (Gemini or OpenAI)
python evaluation/run_ragas_eval.py --provider auto

# Filter evaluation to a specific statute (e.g. Arbitration Law)
python evaluation/run_ragas_eval.py --statute arbitration --offline

# Fast sample evaluation (e.g. first 5 cases)
python evaluation/run_ragas_eval.py --sample 5 --offline
```

**Official RAGAS Benchmark Scorecard (35 Cases)**:
| RAGAS Metric | Overall Score | Egyptian Civil Code (25 Cases) | Egyptian Arbitration Law (10 Cases) | Definition & Statutory Interpretation |
|---|---|---|---|---|
| **Faithfulness** | **93.7%** | 91.2% | **100.0%** | Claims directly grounded in retrieved statutory text; zero tolerance for hallucination |
| **Answer Relevancy** | **69.2%** | 66.5% | **75.9%** | Direct pertinence to the specific legal inquiry asked |
| **Context Precision** | **98.1%** | 97.3% | **100.0%** | Authoritative provisions prioritized at rank 1 in candidate retrieval |
| **Context Recall** | **94.3%** | 92.0% | **100.0%** | Full coverage of all mandatory statutory provisions needed for complete advisory |

*Reports are automatically serialized to `evaluation/ragas_report.json` and summarized in `evaluation/ragas_summary.md`.*

### Run Unit Tests
```bash
python -m pytest tests/ -v
```
*32/32 Unit Tests passing (including multi-turn chat pipeline, streaming token generation, RAGAS adapter, and full statutory suites).*

### Clear Cache
To flush the Redis/in-memory semantic cache, `__pycache__`, and test caches:
```bash
python scripts/clear_cache.py
```

---

## Project Structure

```text
legal-rag/
├── app/
│   └── streamlit_app.py           # Conversational Chat UI with streaming tokens, statute switcher & citations
├── src/
│   └── legal_rag/
│       ├── __init__.py
│       ├── config.py              # Pydantic application settings (gemini-3.1-flash-lite default)
│       ├── models.py              # Domain contracts (LegalArticle, RAGResponse, etc.)
│       ├── pipeline.py            # RAGService orchestrator with streaming & conversational context
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── normalizer.py      # Arabic normalization & OCR ligature repair engine
│       │   ├── parser.py          # Positional PyMuPDF two-column bilingual parser
│       │   └── validator.py       # Article range & integrity validator
│       ├── retrieval/
│       │   ├── __init__.py
│       │   ├── query_parser.py    # Language & deterministic article extractor (Western & Indic, cached)
│       │   ├── embeddings.py      # BGE-M3 dense embedding model wrapper (LRU cached)
│       │   ├── qdrant.py          # Versioned Qdrant store adapter with namespaced UUIDs & memory replica
│       │   ├── sparse.py          # BM25Plus sparse lexical retriever
│       │   ├── hybrid.py          # RRF fusion & exact priority injection
│       │   └── reranker.py        # Dynamic candidate pool cross-encoder reranker
│       ├── generation/
│       │   ├── __init__.py
│       │   ├── grounding_gate.py  # Hard pre-LLM validation gate (multi-statute bounds)
│       │   ├── prompts.py         # Grounding prompt & multi-turn consultation history
│       │   └── llm.py             # Gemini, OpenAI, & Mock LLM clients with real-time streaming
│       ├── cache/
│       │   ├── __init__.py
│       │   └── semantic_cache.py  # Redis semantic cache with cross-statute namespace isolation
│       └── evaluation/
│           ├── __init__.py
│           └── ragas_adapter.py   # RAGAS metrics adapter, local BGE-M3 embeddings, & offline judge
├── scripts/
│   ├── ingest.py                  # Multi-statute corpus parsing & validation entry point
│   ├── build_index.py             # Multi-statute embedding & indexing entry point
│   └── clear_cache.py             # Cache administration & cache-flush utility
├── data/
│   ├── corpora_registry.json      # Central registry for cataloged Egyptian statutes
│   ├── corpus_config.json         # Decoupled corpus metadata
│   ├── law_books/                 # Dedicated repository for source bilingual statutory PDFs
│   │   ├── القانون المدني المصري.pdf   # Egyptian Civil Code (Law 131 of 1948)
│   │   └── قانون التحكيم المصري.pdf   # Egyptian Arbitration Law (Law 27 of 1994)
│   ├── processed/
│   │   ├── egyptian_civil_code.json          # 1,149 structured civil code articles
│   │   ├── egyptian_arbitration_law.json     # 58 structured arbitration law articles
│   │   ├── bm25_index.pkl                    # Unified BM25Plus index (1,207 articles)
│   │   ├── bge_m3_embeddings.pkl             # Serialized Civil Code dense vectors
│   │   └── bge_m3_embeddings_arbitration.pkl # Serialized Arbitration Law dense vectors
│   └── qdrant_db/                 # Embedded local Qdrant vector database storage
├── evaluation/
│   ├── questions.json             # 35 curated benchmark cases (25 Civil + 10 Arbitration)
│   ├── run_eval.py                # Automated evaluation runner
│   ├── run_ablation.py            # Retrieval ablation benchmark runner
│   ├── run_ragas_eval.py          # RAGAS benchmark runner (CLI & automated scoring)
│   ├── eval_report.json           # Serialized evaluation report
│   ├── ragas_report.json          # Serialized RAGAS benchmark results
│   └── ragas_summary.md           # Formatted RAGAS scorecard summary
├── tests/                         # Full unit test suite (32 tests including test_chat_pipeline.py)
├── docker-compose.yml             # Docker services for Qdrant & Redis
├── Dockerfile                     # Containerized application build
├── pyproject.toml                 # Package configuration & dependencies (TOML)
├── requirements.txt               # Plain text requirements specification
├── .env.example                   # Environment configuration template
└── README.md                      # Comprehensive project documentation
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
