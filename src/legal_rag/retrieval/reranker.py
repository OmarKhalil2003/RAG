import os
import math
from sentence_transformers import CrossEncoder
from legal_rag.models import RetrievalResult


class LegalReranker:
    """
    Multilingual cross-encoder reranker for legal query-passage pairs.
    Uses BAAI/bge-reranker-v2-m3 or local cross-encoder model.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self.model: CrossEncoder | None = None
        self._load_failed = False

    def _get_model(self) -> CrossEncoder | None:
        if self.model is None and not self._load_failed:
            try:
                import torch
                torch.set_num_threads(os.cpu_count() or 4)
                self.model = CrossEncoder(self.model_name, max_length=128)
            except Exception as e:
                print(f"[LegalReranker] Could not load CrossEncoder {self.model_name}: {e}. Using score-based fallback.")
                self._load_failed = True
        return self.model

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int = 5
    ) -> list[RetrievalResult]:
        """Rerank candidates using cross-encoder relevance scoring."""
        if not candidates:
            return []

        # Exact article matches always preserve top priority
        exact_matches = [c for c in candidates if "exact" in c.sources or c.score >= 1.0]
        other_candidates = [c for c in candidates if "exact" not in c.sources and c.score < 1.0]

        model = self._get_model()
        if model is not None and other_candidates:
            # Dynamic reranker pool: If exact matches are already secured at rank 1,
            # rerank only top 3 non-exact candidates. Otherwise rerank top 5 to minimize CPU latency.
            pool_size = 3 if exact_matches else 5
            rerank_pool = other_candidates[:pool_size]
            pairs = [
                (query, f"المادة {c.article_number} (Article {c.article_number})\n{c.text_ar[:200]}\n{c.text_en[:200]}")
                for c in rerank_pool
            ]
            scores = model.predict(pairs, batch_size=pool_size, show_progress_bar=False)

            scored_candidates = []
            for c, s in zip(rerank_pool, scores):
                c_copy = c.model_copy()
                val = float(s)
                # Calibrate raw logit to probability in [0, 1] via sigmoid
                prob = 1.0 / (1.0 + math.exp(-max(min(val, 20.0), -20.0)))
                c_copy.score = float(prob)
                c_copy.sources = list(set(c_copy.sources + ["reranker"]))
                scored_candidates.append(c_copy)

            scored_candidates.sort(key=lambda x: x.score, reverse=True)
            # Combine exact matches first, then reranked candidates
            combined = exact_matches + scored_candidates + other_candidates[pool_size:]
            return combined[:top_k]
        else:
            # Fallback: retain candidate ranking (exact priority + RRF)
            for c in candidates:
                if "reranker" not in c.sources:
                    c.sources.append("reranker")
            return candidates[:top_k]
