import sys
import json
import os
import logging
from pathlib import Path
import streamlit as st

# Silence noisy file watcher logs on third-party libraries (e.g. transformers torchvision check)
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

# Setup python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from legal_rag.config import settings
from legal_rag.models import UserRole
from legal_rag.pipeline import RAGService
from legal_rag.generation.llm import GeminiLLMClient, OpenAILLMClient, MockLLMClient
from legal_rag.ingestion.normalizer import clean_arabic_ocr_artifacts

# Page configuration
st.set_page_config(
    page_title="JURIS-EGYPT | Statutory Legal Intelligence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Professional Enterprise Legal Styling (Dark Mode & Light Mode Compatible)
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700;800&family=Inter:wght@400;500;600;700&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;1,6..72,400&family=Amiri:ital,wght@0,400;0,700;1,400&display=swap" rel="stylesheet">

<style>
    /* Global Base */
    html, body {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Institutional Masthead */
    .masthead-container {
        background: linear-gradient(135deg, #0B132B 0%, #1C2541 60%, #1E293B 100%);
        padding: 24px 32px;
        border-radius: 12px;
        margin-bottom: 24px;
        border: 1px solid #334155;
        box-shadow: 0 4px 20px -2px rgba(11, 19, 43, 0.25);
    }
    .masthead-brand {
        font-family: 'Cinzel', serif;
        font-size: 1.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        color: #F8FAFC;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .masthead-brand span {
        color: #D97706;
    }
    .masthead-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        font-weight: 400;
        color: #94A3B8;
        margin-bottom: 6px;
        letter-spacing: 0.02em;
    }

    /* Section Typography - Dark & Light Mode Compatible */
    .section-label {
        font-family: 'Cinzel', serif;
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        color: inherit !important;
        text-transform: uppercase;
        border-bottom: 2px solid rgba(128, 128, 128, 0.2);
        padding-bottom: 8px;
        margin-top: 24px;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .section-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        font-weight: 500;
        color: inherit !important;
        opacity: 0.75;
        text-transform: none;
        letter-spacing: normal;
    }

    /* Legal Opinion Dossier - Adaptive Theme */
    .dossier-card {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 20px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
        color: inherit !important;
    }
    .dossier-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
        padding-bottom: 12px;
        margin-bottom: 16px;
    }
    .dossier-title {
        font-family: 'Cinzel', serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: inherit !important;
        letter-spacing: 0.04em;
    }
    .dossier-persona {
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        font-weight: 600;
        color: inherit !important;
        background: rgba(128, 128, 128, 0.18);
        padding: 4px 10px;
        border-radius: 6px;
    }
    .dossier-body {
        font-family: 'Newsreader', Georgia, serif;
        font-size: 1.12rem;
        line-height: 1.8;
        color: inherit !important;
    }
    .dossier-body * {
        color: inherit !important;
    }
    .dossier-body-ar {
        font-family: 'Amiri', serif;
        font-size: 1.28rem;
        line-height: 1.95;
        direction: rtl;
        text-align: right;
        color: inherit !important;
    }
    .dossier-body-ar * {
        color: inherit !important;
    }

    /* Telemetry Audit Strip */
    .audit-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        padding: 10px 18px;
        margin-bottom: 20px;
        font-size: 0.82rem;
        color: inherit !important;
        opacity: 0.88;
    }
    .audit-item {
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .audit-item strong {
        color: inherit !important;
        opacity: 1.0;
    }
    .status-active {
        color: #10B981 !important;
        font-weight: 600;
    }
    .status-cached {
        color: #3B82F6 !important;
        font-weight: 600;
    }

    /* Statutory Provisions Display */
    .arabic-statute-text {
        font-family: 'Amiri', serif;
        font-size: 1.18rem;
        line-height: 1.85;
        direction: rtl;
        text-align: right;
        color: inherit !important;
        background: rgba(128, 128, 128, 0.12);
        padding: 16px 20px;
        border-radius: 8px;
        border: 1px solid rgba(128, 128, 128, 0.25);
    }
    .arabic-statute-text * {
        color: inherit !important;
    }
    .english-statute-text {
        font-family: 'Newsreader', Georgia, serif;
        font-size: 1.05rem;
        line-height: 1.65;
        color: inherit !important;
        background: rgba(128, 128, 128, 0.12);
        padding: 16px 20px;
        border-radius: 8px;
        border: 1px solid rgba(128, 128, 128, 0.25);
    }
    .english-statute-text * {
        color: inherit !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_rag_service():
    return RAGService()


@st.cache_data
def get_corpora_registry():
    reg_file = project_root / "data" / "corpora_registry.json"
    if reg_file.exists():
        with open(reg_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return [
        {
            "id": "civil",
            "document_id": "egyptian_civil_code_1948",
            "jurisdiction": "Egypt",
            "law_type": "civil",
            "law_name_ar": "القانون المدني المصري",
            "law_name_en": "Egyptian Civil Code",
            "law_year": 1948,
            "law_number": "Law No. 131 of 1948",
            "source_pdf": "data/law_books/القانون المدني المصري.pdf",
            "processed_file": "data/processed/egyptian_civil_code.json",
            "expected_article_range": [1, 1149],
            "total_articles": 1149,
            "repealed_count": 56,
            "description_ar": "القانون العام للمعاملات والالتزامات والحقوق العينية والشخصية",
            "description_en": "General code governing obligations, contracts, civil liability, and real rights"
        },
        {
            "id": "arbitration",
            "document_id": "egyptian_arbitration_law_1994",
            "jurisdiction": "Egypt",
            "law_type": "arbitration",
            "law_name_ar": "قانون التحكيم المصري",
            "law_name_en": "Egyptian Arbitration Law",
            "law_year": 1994,
            "law_number": "Law No. 27 of 1994",
            "source_pdf": "data/law_books/قانون التحكيم المصري.pdf",
            "processed_file": "data/processed/egyptian_arbitration_law.json",
            "expected_article_range": [1, 58],
            "total_articles": 58,
            "repealed_count": 0,
            "description_ar": "قانون التحكيم في المواد المدنية والتجارية الداخلية والدولية",
            "description_en": "Law on arbitration in civil and commercial matters, domestic and international"
        }
    ]


@st.cache_resource
def ensure_all_corpora_indexed(_rag_service, corpora):
    """Ensures all cataloged statutes have their articles indexed in Qdrant and unified BM25 index loaded."""
    try:
        import pickle
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue
        from legal_rag.models import LegalArticle
        from legal_rag.retrieval.sparse import BM25SparseRetriever

        # Refresh BM25 index if not containing full multi-statute corpus
        if len(_rag_service.sparse_retriever.articles) < 1200:
            bm25_p = project_root / "data" / "processed" / "bm25_index.pkl"
            if bm25_p.exists():
                _rag_service.sparse_retriever = BM25SparseRetriever.load(bm25_p)

        # Verify Qdrant points for each statute
        for corp in corpora:
            law_type = corp["law_type"]
            flt = Filter(must=[FieldCondition(key="law_type", match=MatchValue(value=law_type))])
            try:
                cnt = _rag_service.qdrant_store.client.count(
                    collection_name=_rag_service.qdrant_store.collection_name,
                    count_filter=flt,
                    exact=True
                ).count
            except Exception:
                cnt = 0

            if cnt < corp["total_articles"]:
                proc_file = project_root / corp["processed_file"]
                emb_file = project_root / (
                    "data/processed/bge_m3_embeddings_arbitration.pkl"
                    if law_type == "arbitration"
                    else "data/processed/bge_m3_embeddings.pkl"
                )
                if proc_file.exists() and emb_file.exists():
                    with open(proc_file, "r", encoding="utf-8") as f:
                        raw_data = json.load(f)
                    articles = [LegalArticle(**item) for item in raw_data]
                    with open(emb_file, "rb") as f:
                        vectors = pickle.load(f)
                    _rag_service.qdrant_store.index_articles(articles, vectors)
                    logging.info(f"Indexed {len(articles)} articles of {corp['law_name_en']} into Qdrant.")
    except Exception as e:
        logging.warning(f"Corpora index verification notice: {e}")


corpora = get_corpora_registry()
rag_service = get_rag_service()
ensure_all_corpora_indexed(rag_service, corpora)


# --- SIDEBAR: EXECUTIVE CONTROL PANEL ---
with st.sidebar:
    st.markdown("### Institutional Configuration")

    # 1. Governing Statute Selection
    st.markdown("#### 🏛️ Governing Statute / التقنين التشريعي")
    statute_options = {
        f"📜 {c['law_name_en']} ({c['law_name_ar']}) — {c['law_number']}": c["id"]
        for c in corpora
    }

    if "current_statute_id" not in st.session_state:
        st.session_state["current_statute_id"] = "civil"

    statute_ids = list(statute_options.values())
    curr_idx = statute_ids.index(st.session_state["current_statute_id"]) if st.session_state["current_statute_id"] in statute_ids else 0

    selected_statute_label = st.selectbox(
        "Select Active Statute:",
        list(statute_options.keys()),
        index=curr_idx,
        help="Select the legislative statute to interrogate. Cross-statutory isolation is strictly enforced."
    )
    new_statute_id = statute_options[selected_statute_label]
    if new_statute_id != st.session_state["current_statute_id"]:
        st.session_state["current_statute_id"] = new_statute_id
        if new_statute_id == "arbitration":
            st.session_state["query_input"] = "ما هي الشروط الشكلية والموضوعية لصحة اتفاق التحكيم وفقاً للمادة 10 من قانون التحكيم المصري؟"
        else:
            st.session_state["query_input"] = "ما هي أحكام المادة 157 من القانون المدني المصري وحالات فسخ العقد والتعويض عند الإخلال بالالتزام؟"
        st.rerun()

    active_statute = next((c for c in corpora if c["id"] == st.session_state["current_statute_id"]), corpora[0])
    st.caption(f"📌 **Focus:** {active_statute['description_en']}")

    st.markdown("---")
    
    # 2. Advisory Profile Persona
    role_options = {
        "Legal Counsel / Advocate (محامٍ)": UserRole.LAWYER,
        "General Advisory / Citizen (مواطن)": UserRole.CITIZEN,
        "Academic / Law Student (طالب حقوق)": UserRole.LAW_STUDENT,
        "Legislative Researcher (باحث قانوني)": UserRole.LEGAL_RESEARCHER,
    }
    selected_role_label = st.selectbox(
        "Advisory Profile:",
        list(role_options.keys()),
        index=0,
        help="Adjusts the presentation style and doctrinal depth of the generated opinion while preserving strict evidence grounding."
    )
    user_role = role_options[selected_role_label]

    role_descriptions = {
        UserRole.LAWYER: "Formal memorandum style, direct statutory citation, elements and contractual remedies.",
        UserRole.CITIZEN: "Clear plain-language breakdown of rights and liabilities with direct statutory citations.",
        UserRole.LAW_STUDENT: "Doctrinal analysis, rule rationale, legal principles, and statutory construction.",
        UserRole.LEGAL_RESEARCHER: "Systematic statutory exegesis cross-referencing code books, chapters, and sections.",
    }
    st.caption(role_descriptions[user_role])

    st.markdown("---")
    st.markdown("### LLM Intelligence Engine")
    
    # Provider selection
    llm_providers = [
        "Google Gemini",
        "OpenAI",
        "Deterministic Engine (Offline Grounded Mock)"
    ]
    # Default to Gemini if configured
    default_idx = 0 if "gemini" in settings.llm_provider.lower() else (1 if "openai" in settings.llm_provider.lower() else 2)
    selected_provider_label = st.selectbox("Provider:", llm_providers, index=default_idx)

    custom_llm_client = None
    if "Gemini" in selected_provider_label:
        gemini_model_options = [
            "gemini-3.1-flash-lite",
            "gemini-3-flash-preview",
            "gemini-flash-latest",
            "gemini-flash-lite-latest",
            "gemini-3.5-flash-lite",
            "gemini-3.5-flash",
            "Custom Model Name..."
        ]
        selected_choice = st.selectbox(
            "Gemini Model Edition:",
            gemini_model_options,
            index=0,
            help="gemini-3.1-flash-lite and gemini-3-flash-preview offer the fastest speed and highest quota availability."
        )
        if selected_choice == "Custom Model Name...":
            selected_model = st.text_input("Enter Model Identifier:", value="gemini-3.1-flash-lite")
        else:
            selected_model = selected_choice
        
        env_key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        api_key_val = st.text_input(
            "Gemini API Key:",
            value=env_key,
            type="password",
            help="Enter your Google AI Studio API key or configure GEMINI_API_KEY in .env"
        )
        if api_key_val:
            custom_llm_client = GeminiLLMClient(api_key=api_key_val, model_name=selected_model)
            st.success(f"● Connected · {selected_model}")
        else:
            st.info("● Running via deterministic test generator until Gemini key is entered.")
            custom_llm_client = MockLLMClient()

    elif "OpenAI" in selected_provider_label:
        openai_models = ["gpt-4o-mini", "gpt-4o"]
        selected_oa_model = st.selectbox("Model Edition:", openai_models, index=0)
        env_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY", "")
        openai_key = st.text_input("OpenAI API Key:", value=env_key, type="password")
        if openai_key:
            custom_llm_client = OpenAILLMClient(api_key=openai_key, model=selected_oa_model)
            st.success(f"● Connected · {selected_oa_model}")
        else:
            st.info("● Running via deterministic test generator until OpenAI key is entered.")
            custom_llm_client = MockLLMClient()
    else:
        custom_llm_client = MockLLMClient()
        st.info("● Offline deterministic legal evaluation engine active.")

    st.markdown("---")
    st.markdown("### Cache Administration")
    st.caption(f"Semantic Cache Floor: Cosine Distance ≤ {settings.semantic_cache_distance_threshold:.2f}")
    if st.button("Flush Semantic Cache", use_container_width=True):
        rag_service.cache.clear()
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Semantic cache and session state flushed.")

    with st.expander("⚙️ System Specifications & Architecture", expanded=False):
        st.markdown(f"**Jurisdiction:** {active_statute.get('jurisdiction', 'Egypt')}")
        st.markdown(f"**Active Statute:** {active_statute.get('law_name_en')}")
        st.markdown(f"**Arabic Title:** {active_statute.get('law_name_ar')}")
        st.markdown(f"**Enacted:** {active_statute.get('law_number')}")
        st.markdown(f"**Articles Cataloged:** {active_statute.get('total_articles')} Articles")
        st.markdown(f"**Repealed Provisions Flagged:** {active_statute.get('repealed_count', 0)} articles")
        st.markdown(f"**Multi-Statute Index:** 1,207 articles indexed across {len(corpora)} statutes")
        st.markdown(f"**Vector Store:** Qdrant (`{settings.qdrant_collection_name}`)")
        st.markdown(f"**Dense Embedding:** BAAI/bge-m3 (1024-dim)")
        st.markdown(f"**Sparse Index:** BM25Plus Lexical")
        st.markdown(f"**Cross-Encoder:** BAAI/bge-reranker-v2-m3")
        st.markdown(f"**Statutory Isolation:** Isolated by `law_type='{active_statute['law_type']}'`")

    with st.expander("📊 RAGAS Evaluation Benchmarks", expanded=False):
        ragas_report_file = project_root / "evaluation" / "ragas_report.json"
        if ragas_report_file.exists():
            try:
                with open(ragas_report_file, "r", encoding="utf-8") as f:
                    ragas_data = json.load(f)
                agg = ragas_data.get("aggregate_scores", {})
                st.markdown(f"**Evaluator Engine:** `{ragas_data.get('provider', 'N/A').upper()}`")
                st.markdown(f"**Total Benchmark Cases:** `{ragas_data.get('total_cases', 0)}`")
                
                c_m1, c_m2 = st.columns(2)
                c_m1.metric("Faithfulness", f"{agg.get('faithfulness', 0.0) * 100:.1f}%")
                c_m2.metric("Relevancy", f"{agg.get('answer_relevancy', 0.0) * 100:.1f}%")
                
                c_m3, c_m4 = st.columns(2)
                c_m3.metric("Precision", f"{agg.get('context_precision', 0.0) * 100:.1f}%")
                c_m4.metric("Recall", f"{agg.get('context_recall', 0.0) * 100:.1f}%")
                
                breakdown = ragas_data.get("statute_breakdown", {})
                if active_statute["law_type"] in breakdown:
                    st_scores = breakdown[active_statute["law_type"]]
                    st.caption(f"**Active Statute ({active_statute['law_name_en']}):**")
                    st.caption(f"Faithfulness: {st_scores['faithfulness']*100:.1f}% · Recall: {st_scores['context_recall']*100:.1f}%")
            except Exception as e:
                st.caption(f"Report loading notice: {e}")
        else:
            st.caption("No RAGAS report found. Run `python evaluation/run_ragas_eval.py` to generate.")

        if st.button("Run Quick RAGAS Audit (Sample)", use_container_width=True):
            with st.spinner("Running live RAGAS evaluation sample..."):
                from legal_rag.evaluation.ragas_adapter import prepare_sample, execute_ragas_evaluation
                sample_cases = [
                    {
                        "question": "ما هي أحكام المادة 147 من القانون المدني؟",
                        "expected_law_type": "civil",
                        "expected_jurisdiction": "Egypt",
                        "expected_articles": [147],
                        "ground_truth": "العقد شريعة المتعاقدين وتطبق نظرية الظروف الطارئة."
                    },
                    {
                        "question": "ما هو نطاق سريان قانون التحكيم المصري؟",
                        "expected_law_type": "arbitration",
                        "expected_jurisdiction": "Egypt",
                        "expected_articles": [1],
                        "ground_truth": "يسري على كل تحكيم بين أشخاص القانون العام أو الخاص في مصر."
                    }
                ]
                live_samples = []
                for sc in sample_cases:
                    resp = rag_service.ask(
                        question=sc["question"],
                        role=user_role,
                        jurisdiction=sc["expected_jurisdiction"],
                        law_type=sc["expected_law_type"]
                    )
                    live_samples.append(prepare_sample(
                        question=sc["question"],
                        response=resp,
                        ground_truth=sc["ground_truth"],
                        metadata=sc
                    ))
                live_eval = execute_ragas_evaluation(live_samples, provider="mock")
                st.success(f"Audit Complete ({live_eval['provider']}): Faithfulness: {live_eval['aggregate_scores']['faithfulness']*100:.1f}% | Precision: {live_eval['aggregate_scores']['context_precision']*100:.1f}% | Recall: {live_eval['aggregate_scores']['context_recall']*100:.1f}%")



# --- MAIN INTERFACE: LEGAL MASTHEAD ---
st.markdown(f"""
<div class="masthead-container">
    <div class="masthead-brand">
        JURIS-EGYPT <span>·</span> STATUTORY INTELLIGENCE
    </div>
    <div class="masthead-subtitle">
        Bilingual Legal Research & Statutory Grounding Platform · {active_statute['law_name_en']} ({active_statute['law_number']})
    </div>
</div>
""", unsafe_allow_html=True)


# --- STATUTORY QUERY PRESETS ---
st.markdown(f"<div class='section-label'>Statutory Reference Docket — {active_statute['law_name_en']} <span class='section-subtitle'>Select a curated docket case or enter a bespoke statutory inquiry</span></div>", unsafe_allow_html=True)

# State initialization
if "query_input" not in st.session_state:
    if active_statute["id"] == "arbitration":
        st.session_state["query_input"] = "ما هي الشروط الشكلية والموضوعية لصحة اتفاق التحكيم وفقاً للمادة 10 من قانون التحكيم المصري؟"
    else:
        st.session_state["query_input"] = "ما هي أحكام المادة 157 من القانون المدني المصري وحالات فسخ العقد والتعويض عند الإخلال بالالتزام؟"

if "trigger_analysis" not in st.session_state:
    st.session_state["trigger_analysis"] = False

col_d1, col_d2, col_d3, col_d4, col_d5, col_d6 = st.columns(6)

if active_statute["id"] == "arbitration":
    if col_d1.button("Art. 1 (Scope)", help="Scope of application of Law 27/1994", use_container_width=True):
        st.session_state["query_input"] = "ما هو نطاق سريان قانون التحكيم المصري رقم 27 لسنة 1994 وفقاً للمادة 1؟"
        st.session_state["trigger_analysis"] = True
        st.rerun()

    if col_d2.button("المادة ١٠ (Indic)", help="Article 10: Arbitration agreement requirements", use_container_width=True):
        st.session_state["query_input"] = "ما هي الشروط الشكلية والموضوعية لصحة اتفاق التحكيم وفقاً للمادة ١٠ من قانون التحكيم؟"
        st.session_state["trigger_analysis"] = True
        st.rerun()

    if col_d3.button("Art. 22 (Competence)", help="Article 22: Competence-Competence principle", use_container_width=True):
        st.session_state["query_input"] = "ما هي سلطة هيئة التحكيم في الفصل في اختصاصها ومبدأ استقلال شرط التحكيم وفقاً للمادة 22؟"
        st.session_state["trigger_analysis"] = True
        st.rerun()

    if col_d4.button("Art. 39 (Governing Law)", help="Article 39: Substantive governing law", use_container_width=True):
        st.session_state["query_input"] = "ما هو القانون الواجب التطبيق على موضوع النزاع في التحكيم وفقاً للمادة 39؟"
        st.session_state["trigger_analysis"] = True
        st.rerun()

    if col_d5.button("Art. 53 (Nullity)", help="Article 53: Actions for nullity of arbitral awards", use_container_width=True):
        st.session_state["query_input"] = "ما هي حالات وإجراءات رفع دعوى بطلان حكم التحكيم وفقاً للمادة 53 من قانون التحكيم؟"
        st.session_state["trigger_analysis"] = True
        st.rerun()

    if col_d6.button("Penal (Out of Scope)", help="Test domain gate refusal on criminal law query", use_container_width=True):
        st.session_state["query_input"] = "ما هي عقوبة السرقة بالإكراه في القانون المصري؟"
        st.session_state["trigger_analysis"] = True
        st.rerun()

else:
    if col_d1.button("Art. 147 (Western)", help="Article 147: Western numeral inquiry", use_container_width=True):
        st.session_state["query_input"] = "ما هي أحكام المادة 147 من القانون المدني المصري ونظرية الظروف الطارئة؟"
        st.session_state["trigger_analysis"] = True
        st.rerun()

    if col_d2.button("المادة ١٤٧ (Indic)", help="Article 147: Arabic-Indic numeral inquiry", use_container_width=True):
        st.session_state["query_input"] = "ما هي أحكام المادة ١٤٧ من القانون المدني المصري ونظرية الظروف الطارئة؟"
        st.session_state["trigger_analysis"] = True
        st.rerun()

    if col_d3.button("Art. 157 (Breach)", help="Article 157: Bilateral contract breach & rescission", use_container_width=True):
        st.session_state["query_input"] = "ما هي أحكام المادة 157 من القانون المدني المصري وحالات فسخ العقد والتعويض عند الإخلال بالالتزام؟"
        st.session_state["trigger_analysis"] = True
        st.rerun()

    if col_d4.button("Art. 148 (Good Faith)", help="Article 148: Good faith performance", use_container_width=True):
        st.session_state["query_input"] = "ما هي أحكام المادة 148 من القانون المدني وقواعد تنفيذ العقود وفقاً لمبدأ حسن النية؟"
        st.session_state["trigger_analysis"] = True
        st.rerun()

    if col_d5.button("Art. 54 (Repealed)", help="Repealed provision status verification", use_container_width=True):
        st.session_state["query_input"] = "ما هو الوضع القانوني الحالي للمادة 54 من القانون المدني المصري؟"
        st.session_state["trigger_analysis"] = True
        st.rerun()

    if col_d6.button("Penal (Out of Scope)", help="Test domain gate refusal on criminal law query", use_container_width=True):
        st.session_state["query_input"] = "ما هي عقوبة القتل العمد مع سبق الإصرار في القانون المصري؟"
        st.session_state["trigger_analysis"] = True
        st.rerun()


# --- QUERY SEARCH INPUT ---
query_text = st.text_input(
    "Statutory Inquiry / Legal Query:",
    value=st.session_state["query_input"],
    placeholder=f"Enter statutory article, legal principle, or substantive query for {active_statute['law_name_en']} in Arabic or English...",
    label_visibility="collapsed"
)
st.session_state["query_input"] = query_text

btn_col, _ = st.columns([3, 7])
manual_execute = btn_col.button("Conduct Legal Analysis →", type="primary", use_container_width=True)

should_run = manual_execute or st.session_state.get("trigger_analysis", False)
# Clear trigger so subsequent reruns don't loop
st.session_state["trigger_analysis"] = False

if should_run and query_text.strip():
    # Inject user-selected LLM client
    rag_service.llm_client = custom_llm_client

    with st.spinner("Executing hybrid retrieval, reranking statutory candidates, and validating legal grounding..."):
        response = rag_service.ask(
            question=query_text.strip(),
            role=user_role,
            jurisdiction=active_statute.get("jurisdiction", "Egypt"),
            law_type=active_statute.get("law_type", "civil")
        )

    # Detect Arabic for typography
    is_arabic_query = response.query_signals.language == "ar"

    # --- TELEMETRY AUDIT STRIP ---
    cache_badge = '<span class="status-cached">⚡ SEMANTIC CACHE HIT</span>' if response.cached else '<span class="status-active">● RETRIEVED FROM CORPUS</span>'
    st.markdown(f"""
    <div class="audit-strip">
        <div class="audit-item"><strong>Audit Status:</strong> {cache_badge}</div>
        <div class="audit-item"><strong>Execution Latency:</strong> {response.latency_ms:.1f} ms</div>
        <div class="audit-item"><strong>Authorities Cited:</strong> {response.retrieval_count} Provisions</div>
        <div class="audit-item"><strong>Query Language:</strong> {response.query_signals.language.upper()}</div>
        <div class="audit-item"><strong>Grounding Gate:</strong> {'VERIFIED COMPLIANT' if response.sources else 'REFUSAL ENFORCED'}</div>
    </div>
    """, unsafe_allow_html=True)

    # --- LEGAL OPINION DOSSIER ---
    persona_labels = {
        UserRole.LAWYER: "Formal Legal Counsel Memorandum",
        UserRole.CITIZEN: "Client Advisory Summary",
        UserRole.LAW_STUDENT: "Doctrinal Analysis & Case Commentary",
        UserRole.LEGAL_RESEARCHER: "Systematic Legislative Exegesis",
    }
    dossier_persona_name = persona_labels.get(user_role, "Legal Memorandum")
    
    body_class = "dossier-body-ar" if is_arabic_query else "dossier-body"
    
    # Build compact citation badges
    citation_badges = []
    if response.sources:
        for s in response.sources:
            if s.is_repealed:
                badge = f'<span style="background: #FEE2E2; color: #991B1B; border: 1px solid #FCA5A5; font-size: 0.78rem; font-weight: 600; padding: 4px 10px; border-radius: 4px; display: inline-block;">📜 {s.citation} (REPEALED / ملغاة)</span>'
            else:
                badge = f'<span style="background: #EFF6FF; color: #1E40AF; border: 1px solid #BFDBFE; font-size: 0.78rem; font-weight: 600; padding: 4px 10px; border-radius: 4px; display: inline-block;">📜 {s.citation} (ACTIVE)</span>'
            citation_badges.append(badge)
    badges_html = " ".join(citation_badges)

    citations_section = ""
    if badges_html:
        citations_section = f'<div style="margin-top: 20px; padding-top: 14px; border-top: 1px solid #E2E8F0; display: flex; flex-wrap: wrap; gap: 8px; align-items: center;"><span style="font-family: \'Inter\', sans-serif; font-size: 0.8rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.04em;">Referenced Provisions:</span> {badges_html}</div>'

    # Note: Column-0 alignment ensures Markdown parser does not treat HTML as indented code blocks (<pre><code>)
    dossier_html = f"""<div class="dossier-card">
<div class="dossier-header">
<div class="dossier-title">Legal Opinion & Substantive Analysis</div>
<div class="dossier-persona">{dossier_persona_name}</div>
</div>
<div class="{body_class}">

{response.answer}

</div>
{citations_section}
</div>"""
    st.markdown(dossier_html, unsafe_allow_html=True)


    # --- OPTIONAL COLLAPSED STATUTORY PROVISIONS (CLEAN & NON-INTRUSIVE) ---
    if response.sources:
        with st.expander(f"📖 Inspect Referenced Statutory Provisions ({len(response.sources)} Articles Cited)", expanded=False):
            for idx, src in enumerate(response.sources, 1):
                clean_ar_text = clean_arabic_ocr_artifacts(src.text_ar) if src.text_ar else "النص غير متاح"
                clean_en_text = src.text_en if src.text_en else "Translation not available"
                clean_hier = clean_arabic_ocr_artifacts(src.hierarchy) if src.hierarchy else active_statute["law_name_en"]
                
                title_suffix = " · [REPEALED / ملغاة]" if src.is_repealed else ""
                st.markdown(f"**Authority {idx}: {src.citation}{title_suffix}**")
                st.caption(f"🏛️ {clean_hier}")
                
                if src.is_repealed:
                    st.warning("⚠️ STATUTORY NOTICE: This provision was repealed by subsequent legislation and is no longer active law. (تنبيه تشريعي: ألغيت هذه المادة بموجب تشريع لاحق ولا تعد حكماً سارياً).")
                
                tab_ar, tab_en = st.tabs(["Official Arabic Text (النص الرسمي)", "English Comparative Text"])
                with tab_ar:
                    st.markdown(f"<div class='arabic-statute-text'>{clean_ar_text}</div>", unsafe_allow_html=True)
                with tab_en:
                    st.markdown(f"<div class='english-statute-text'>{clean_en_text}</div>", unsafe_allow_html=True)
                
                if idx < len(response.sources):
                    st.markdown("<div style='margin-bottom: 16px; border-bottom: 1px solid #F1F5F9;'></div>", unsafe_allow_html=True)
    else:
        st.info(f"No statutory authorities cited. The inquiry was determined to be outside the jurisdiction or scope of {active_statute['law_name_en']}.")

    # --- PROCEDURAL & DIAGNOSTIC AUDIT (COLLAPSED BY DEFAULT) ---
    with st.expander("🛠️ Grounding Verification Diagnostics (Audit Log)", expanded=False):
        c_p1, c_p2 = st.columns(2)
        with c_p1:
            st.markdown(f"**Substantive Law Scope:** `{active_statute.get('law_name_en')}` ({active_statute.get('law_number')})")
            st.markdown(f"**Jurisdiction:** `{active_statute.get('jurisdiction')}`")
            st.markdown(f"**Statute Partition (law_type):** `{active_statute.get('law_type')}`")
            st.markdown(f"**Active Advisory Profile:** `{user_role.value}`")
        with c_p2:
            st.markdown(f"**Detected Article References:** `{response.query_signals.article_numbers}`")
            st.markdown(f"**Normalized Ingestion Query:** `{response.query_signals.normalized_query}`")
            st.markdown(f"**Semantic Distance Floor:** `{settings.semantic_cache_distance_threshold}`")
            st.markdown(f"**Grounding Gate Evaluation:** `{'PASSED - Statutorily Grounded' if response.sources else 'REFUSED - Scope or Relevance Boundary'}`")
            st.markdown(f"**Active Vector Store Collection:** `{settings.qdrant_collection_name}`")


