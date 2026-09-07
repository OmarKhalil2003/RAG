import sys
import json
import pickle
from pathlib import Path

# Ensure src is in python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from legal_rag.config import settings
from legal_rag.models import LegalArticle
from legal_rag.retrieval.qdrant import QdrantLegalStore
from legal_rag.retrieval.sparse import BM25SparseRetriever
from legal_rag.retrieval.embeddings import BGEM3Embedder


def main():
    print("=" * 60)
    print("Building Multi-Statute Legal RAG Index (BGE-M3 + BM25 + Qdrant)")
    print("=" * 60)

    registry_path = project_root / "data" / "corpora_registry.json"
    if registry_path.exists():
        with open(registry_path, "r", encoding="utf-8") as f:
            corpora = json.load(f)
    else:
        corpora = [{
            "id": "civil",
            "processed_file": "data/processed/egyptian_civil_code.json",
            "law_name_en": "Egyptian Civil Code"
        }]

    all_articles: list[LegalArticle] = []
    all_dense_vectors: list[list[float]] = []

    embedder = None

    for corpus in corpora:
        corpus_id = corpus.get("id", "law")
        p_rel = corpus.get("processed_file", f"data/processed/{corpus_id}.json")
        p_path = project_root / p_rel
        if not p_path.exists():
            print(f"Warning: {p_path} not found. Run 'python scripts/ingest.py' first.")
            continue

        with open(p_path, "r", encoding="utf-8") as f:
            c_data = json.load(f)
        c_articles = [LegalArticle(**item) for item in c_data]
        print(f"\n[{corpus_id}] Loaded {len(c_articles)} articles from {p_path}")

        # Check for cached embeddings for this specific statute
        cache_name = "bge_m3_embeddings.pkl" if corpus_id == "civil" else f"bge_m3_embeddings_{corpus_id}.pkl"
        emb_file = project_root / "data" / "processed" / cache_name

        if emb_file.exists():
            print(f"[{corpus_id}] Loading cached embeddings from {emb_file}...")
            with open(emb_file, "rb") as f:
                c_vectors = pickle.load(f)
        else:
            print(f"[{corpus_id}] Generating BGE-M3 embeddings for {len(c_articles)} articles...")
            if embedder is None:
                embedder = BGEM3Embedder()
            c_texts = [a.canonical_retrieval_text() for a in c_articles]
            c_vectors = embedder.encode(c_texts, batch_size=32, show_progress_bar=False)
            with open(emb_file, "wb") as f:
                pickle.dump(c_vectors, f)
            print(f"[{corpus_id}] Saved {len(c_vectors)} embeddings to {emb_file}")

        all_articles.extend(c_articles)
        all_dense_vectors.extend(c_vectors)

    print(f"\nTotal Indexed Corpus: {len(all_articles)} articles across {len(corpora)} statutes.")

    # 1. Build and save unified BM25 sparse index
    bm25_file = project_root / "data" / "processed" / "bm25_index.pkl"
    print(f"\nBuilding unified BM25 sparse index...")
    sparse_retriever = BM25SparseRetriever(all_articles)
    sparse_retriever.save(bm25_file)
    print(f"BM25 sparse index saved to: {bm25_file}")

    # 2. Index into Qdrant
    print(f"\nConnecting to Qdrant (Collection: {settings.qdrant_collection_name})...")
    try:
        qdrant_store = QdrantLegalStore()
        indexed_count = qdrant_store.index_articles(all_articles, all_dense_vectors)
        print("-" * 60)
        print("INDEXING REPORT:")
        print(f"Collection Name: {settings.qdrant_collection_name}")
        print(f"Indexed Articles: {indexed_count}")
        print(f"Vector Dimension: {len(all_dense_vectors[0]) if all_dense_vectors else 0}")
        print(f"Qdrant Destination: {settings.qdrant_url or settings.qdrant_storage_path}")
        print(f"BM25 Destination: {bm25_file}")
        print("-" * 60)
        print("Status: INDEX READY")
    except RuntimeError as e:
        if "already accessed by another instance" in str(e):
            print("\n[Notice] Local Qdrant storage is currently accessed by a running process (e.g. Streamlit).")
            print("The BM25 index and embedding caches have been successfully updated.")
            print("To sync vectors to Qdrant, click 'Sync Vector Store' in the Streamlit interface or restart Streamlit.")
        else:
            raise e


if __name__ == "__main__":
    main()
