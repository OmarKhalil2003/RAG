import uuid
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from legal_rag.config import settings
from legal_rag.models import LegalArticle, RetrievalResult


class QdrantLegalStore:
    """
    Qdrant vector store adapter for versioned legal collections.
    Supports both client-server (Docker) and embedded local storage modes.
    """

    def __init__(
        self,
        url: str | None = None,
        storage_path: str | Path | None = None,
        collection_name: str | None = None
    ):
        self.url = url or settings.qdrant_url
        self.storage_path = str(storage_path or settings.qdrant_storage_path)
        self.collection_name = collection_name or settings.qdrant_collection_name
        
        # Initialize client
        if self.url and self.url.startswith("http"):
            try:
                self.client = QdrantClient(url=self.url, timeout=10.0)
                # Quick healthcheck
                self.client.get_collections()
            except Exception:
                # Fallback to local storage if remote Qdrant is unreachable
                Path(self.storage_path).mkdir(parents=True, exist_ok=True)
                self.client = QdrantClient(path=self.storage_path)
        else:
            Path(self.storage_path).mkdir(parents=True, exist_ok=True)
            try:
                self.client = QdrantClient(path=self.storage_path)
            except Exception:
                # Concurrent process fallback (e.g. running Streamlit holds disk lock)
                self.client = QdrantClient(":memory:")
                self._load_memory_replica()

    def _load_memory_replica(self) -> None:
        """Loads cached corpora into an in-memory Qdrant store replica when disk is locked."""
        try:
            import json
            import pickle
            root = Path(__file__).resolve().parent.parent.parent.parent
            reg_path = root / "data" / "corpora_registry.json"
            if not reg_path.exists():
                return
            with open(reg_path, "r", encoding="utf-8") as f:
                corpora = json.load(f)

            for corp in corpora:
                proc_file = root / corp["processed_file"]
                emb_file = root / (
                    "data/processed/bge_m3_embeddings_arbitration.pkl"
                    if corp["law_type"] == "arbitration"
                    else "data/processed/bge_m3_embeddings.pkl"
                )
                if proc_file.exists() and emb_file.exists():
                    with open(proc_file, "r", encoding="utf-8") as f:
                        raw_arts = json.load(f)
                    articles = [LegalArticle(**item) for item in raw_arts]
                    with open(emb_file, "rb") as f:
                        vectors = pickle.load(f)
                    self.index_articles(articles, vectors)
        except Exception:
            pass

    def create_collection_if_not_exists(self, vector_dim: int = 1024) -> None:
        """Create versioned collection with 1024-dim cosine distance if it does not exist."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
            )
            # Create payload indexes for fast filtering
            for field in ["article_number", "jurisdiction", "law_type", "is_repealed"]:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=qmodels.PayloadSchemaType.KEYWORD if field != "article_number" else qmodels.PayloadSchemaType.INTEGER
                )

    def index_articles(
        self,
        articles: list[LegalArticle],
        dense_vectors: list[list[float]],
        batch_size: int = 100
    ) -> int:
        """Index legal articles and their dense vectors into Qdrant."""
        self.create_collection_if_not_exists(vector_dim=len(dense_vectors[0]))
        points = []
        for idx, (art, vec) in enumerate(zip(articles, dense_vectors)):
            payload = art.model_dump()
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{art.document_id}_{art.article_number}"))
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vec,
                    payload=payload
                )
            )
            if len(points) >= batch_size:
                self.client.upsert(collection_name=self.collection_name, points=points)
                points = []

        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)

        return len(articles)

    def _build_filter(
        self,
        jurisdiction: str | None = None,
        law_type: str | None = None,
        article_number: int | None = None
    ) -> Filter | None:
        must_conditions = []
        if jurisdiction:
            must_conditions.append(FieldCondition(key="jurisdiction", match=MatchValue(value=jurisdiction)))
        if law_type:
            must_conditions.append(FieldCondition(key="law_type", match=MatchValue(value=law_type)))
        if article_number is not None:
            must_conditions.append(FieldCondition(key="article_number", match=MatchValue(value=article_number)))

        if not must_conditions:
            return None
        return Filter(must=must_conditions)

    def exact_lookup(
        self,
        article_number: int,
        jurisdiction: str | None = None,
        law_type: str | None = None
    ) -> list[RetrievalResult]:
        """Deterministic exact payload lookup by article number."""
        flt = self._build_filter(jurisdiction=jurisdiction, law_type=law_type, article_number=article_number)
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=flt,
            limit=5,
            with_payload=True,
            with_vectors=False
        )

        results = []
        for p in points:
            pl = p.payload or {}
            doc_id = pl.get("document_id", "art")
            art_num = pl.get("article_number", 0)
            results.append(
                RetrievalResult(
                    chunk_id=f"{doc_id}_{art_num}",
                    article_number=art_num,
                    score=1.0,  # Exact match top priority score
                    sources=["exact"],
                    metadata=pl,
                    text_ar=pl.get("text_ar", ""),
                    text_en=pl.get("text_en", ""),
                    is_repealed=pl.get("is_repealed", False),
                    citation=pl.get("citation", "")
                )
            )
        return results

    def dense_search(
        self,
        query_vector: list[float],
        limit: int = 20,
        jurisdiction: str | None = None,
        law_type: str | None = None
    ) -> list[RetrievalResult]:
        """Dense semantic search using cosine similarity with payload scope filters."""
        flt = self._build_filter(jurisdiction=jurisdiction, law_type=law_type)
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=flt,
                limit=limit,
                with_payload=True
            )
            scored_points = response.points
        else:
            scored_points = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=flt,
                limit=limit,
                with_payload=True
            )

        results = []
        for sp in scored_points:
            pl = sp.payload or {}
            doc_id = pl.get("document_id", "art")
            art_num = pl.get("article_number", 0)
            results.append(
                RetrievalResult(
                    chunk_id=f"{doc_id}_{art_num}",
                    article_number=art_num,
                    score=float(sp.score),
                    sources=["dense"],
                    metadata=pl,
                    text_ar=pl.get("text_ar", ""),
                    text_en=pl.get("text_en", ""),
                    is_repealed=pl.get("is_repealed", False),
                    citation=pl.get("citation", "")
                )
            )
        return results
