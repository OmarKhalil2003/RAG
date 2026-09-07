import json
import logging
import numpy as np
import redis
from pydantic import BaseModel

from legal_rag.config import settings
from legal_rag.models import RAGResponse, UserRole

logger = logging.getLogger("legal_rag.cache")


def cosine_distance(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine distance between two vectors: 1.0 - cosine_similarity."""
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    cos_sim = float(np.dot(a, b) / (norm_a * norm_b))
    return max(0.0, min(2.0, 1.0 - cos_sim))


class RedisSemanticCache:
    """
    Redis-backed semantic response cache with strict legal scope isolation.
    
    Isolation namespace:
        jurisdiction:law_type:user_role:kb_version:prompt_version
        
    Distance threshold:
        Uses cosine distance where lower = stricter semantic match (0.00 to 2.00).
        Matches with distance <= SEMANTIC_CACHE_DISTANCE_THRESHOLD count as CACHE HIT.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        distance_threshold: float | None = None,
        ttl: int | None = None
    ):
        self.redis_url = redis_url or settings.redis_url
        self.distance_threshold = (
            distance_threshold
            if distance_threshold is not None
            else settings.semantic_cache_distance_threshold
        )
        self.ttl = ttl or settings.semantic_cache_ttl
        self.redis_client: redis.Redis | None = None
        self.is_connected = False
        self._local_cache: dict[str, list[dict]] = {}

        self._connect()

    def _connect(self) -> None:
        try:
            self.redis_client = redis.Redis.from_url(
                self.redis_url,
                socket_timeout=2.0,
                decode_responses=False
            )
            self.redis_client.ping()
            self.is_connected = True
            logger.info("Connected to Redis semantic cache at %s", self.redis_url)
        except Exception as e:
            self.is_connected = False
            if settings.local_dev_mode:
                logger.warning("[SemanticCache] Redis offline (%s). Using in-memory cache (LOCAL_DEV_MODE=True).", e)
            else:
                logger.warning("[SemanticCache] Redis offline (%s). Semantic caching DISABLED.", e)

    def _get_namespace(
        self,
        jurisdiction: str,
        law_type: str,
        user_role: UserRole | str
    ) -> str:
        role_str = user_role.value if isinstance(user_role, UserRole) else str(user_role)
        return (
            f"legal_cache:{jurisdiction.lower()}:{law_type.lower()}:{role_str.lower()}:"
            f"{settings.knowledge_base_version}:{settings.prompt_version}"
        )

    def get(
        self,
        query_vector: list[float],
        jurisdiction: str,
        law_type: str,
        user_role: UserRole | str,
        normalized_query: str | None = None,
        article_numbers: list[int] | None = None
    ) -> RAGResponse | None:
        """
        Lookup semantic cache within the specified legal scope namespace.
        Returns cached RAGResponse if:
        1. Exact normalized query match (e.g. Indic vs Western digits normalizes identically).
        2. Direct canonical article query match (same single article inquiry).
        3. Cosine distance <= distance_threshold.
        """
        namespace = self._get_namespace(jurisdiction, law_type, user_role)

        # 1. Try Redis lookup
        if self.is_connected and self.redis_client is not None:
            try:
                keys = self.redis_client.keys(f"{namespace}:*")
                best_distance = float("inf")
                best_response_json = None

                for k in keys:
                    data = self.redis_client.hgetall(k)
                    if not data:
                        continue
                    cached_vec_bytes = data.get(b"vector")
                    cached_resp_bytes = data.get(b"response")
                    cached_norm_bytes = data.get(b"normalized_query")
                    cached_arts_bytes = data.get(b"article_numbers")

                    # Check exact normalized match (Indic vs Western numerals)
                    if normalized_query and cached_norm_bytes:
                        if cached_norm_bytes.decode("utf-8").strip() == normalized_query.strip():
                            resp_dict = json.loads(cached_resp_bytes.decode("utf-8"))
                            resp = RAGResponse(**resp_dict)
                            resp.cached = True
                            return resp

                    # Check canonical single article inquiry match
                    if article_numbers and cached_arts_bytes:
                        try:
                            cached_arts = json.loads(cached_arts_bytes.decode("utf-8"))
                            if len(article_numbers) == 1 and article_numbers == cached_arts:
                                # If both are concise inquiries on the exact same article
                                cached_q = data.get(b"query", b"").decode("utf-8")
                                if len(cached_q.split()) <= 8 and (normalized_query and len(normalized_query.split()) <= 8):
                                    resp_dict = json.loads(cached_resp_bytes.decode("utf-8"))
                                    resp = RAGResponse(**resp_dict)
                                    resp.cached = True
                                    return resp
                        except Exception:
                            pass

                    if cached_vec_bytes and cached_resp_bytes:
                        cached_vec = json.loads(cached_vec_bytes.decode("utf-8"))
                        dist = cosine_distance(query_vector, cached_vec)
                        if dist < best_distance:
                            best_distance = dist
                            best_response_json = cached_resp_bytes.decode("utf-8")

                if best_distance <= self.distance_threshold and best_response_json:
                    resp_dict = json.loads(best_response_json)
                    resp = RAGResponse(**resp_dict)
                    resp.cached = True
                    return resp
            except Exception as e:
                logger.error("[SemanticCache] Redis read error: %s", e)

        # 2. Local dev fallback if Redis offline
        if settings.local_dev_mode:
            entries = self._local_cache.get(namespace, [])
            best_dist = float("inf")
            best_resp = None
            for entry in entries:
                # Check exact normalized match (Indic vs Western numerals)
                if normalized_query and entry.get("normalized_query"):
                    if entry["normalized_query"].strip() == normalized_query.strip():
                        resp = entry["response"].model_copy()
                        resp.cached = True
                        return resp

                # Check canonical single article inquiry match
                if article_numbers and entry.get("article_numbers"):
                    if len(article_numbers) == 1 and article_numbers == entry["article_numbers"]:
                        cached_q = entry.get("query", "")
                        if len(cached_q.split()) <= 8 and (normalized_query and len(normalized_query.split()) <= 8):
                            resp = entry["response"].model_copy()
                            resp.cached = True
                            return resp

                dist = cosine_distance(query_vector, entry["vector"])
                if dist < best_dist:
                    best_dist = dist
                    best_resp = entry["response"]

            if best_dist <= self.distance_threshold and best_resp:
                resp = best_resp.model_copy()
                resp.cached = True
                return resp

        return None

    def store(
        self,
        query: str,
        query_vector: list[float],
        response: RAGResponse,
        jurisdiction: str,
        law_type: str,
        user_role: UserRole | str,
        normalized_query: str | None = None,
        article_numbers: list[int] | None = None
    ) -> bool:
        """
        Store a generated RAGResponse in the semantic cache.
        """
        namespace = self._get_namespace(jurisdiction, law_type, user_role)

        # 1. Try Redis store
        if self.is_connected and self.redis_client is not None:
            try:
                # Key based on query hash to avoid duplicates
                entry_id = abs(hash(query))
                key = f"{namespace}:{entry_id}"
                mapping = {
                    "vector": json.dumps(query_vector),
                    "response": response.model_dump_json(),
                    "query": query,
                    "normalized_query": normalized_query or "",
                    "article_numbers": json.dumps(article_numbers or [])
                }
                self.redis_client.hset(key, mapping=mapping)
                self.redis_client.expire(key, self.ttl)
                return True
            except Exception as e:
                logger.error("[SemanticCache] Redis store error: %s", e)

        # 2. Local dev store
        if settings.local_dev_mode:
            if namespace not in self._local_cache:
                self._local_cache[namespace] = []
            self._local_cache[namespace].append({
                "query": query,
                "vector": query_vector,
                "response": response.model_copy(),
                "normalized_query": normalized_query or "",
                "article_numbers": article_numbers or []
            })
            return True

        return False

    def clear(self) -> None:
        """Clear all entries from both Redis and local in-memory semantic cache."""
        if self.is_connected and self.redis_client is not None:
            try:
                keys = self.redis_client.keys("legal_cache:*")
                if keys:
                    self.redis_client.delete(*keys)
                logger.info("[SemanticCache] Flushed %d keys from Redis.", len(keys))
            except Exception as e:
                logger.error("[SemanticCache] Error clearing Redis keys: %s", e)
        self._local_cache.clear()
        logger.info("[SemanticCache] Cleared in-memory cache.")
