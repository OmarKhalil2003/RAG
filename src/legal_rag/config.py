import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # LLM Settings
    llm_provider: str = Field(default="mock", alias="LLM_PROVIDER")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-3.1-flash-lite", alias="GEMINI_MODEL")

    # Qdrant Settings
    qdrant_url: str | None = Field(default=None, alias="QDRANT_URL")
    qdrant_storage_path: str = Field(default="data/qdrant_db", alias="QDRANT_STORAGE_PATH")

    # Redis Semantic Cache Settings
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    semantic_cache_distance_threshold: float = Field(
        default=0.10,
        alias="SEMANTIC_CACHE_DISTANCE_THRESHOLD",
        description="Cosine distance threshold (lower = stricter match)"
    )
    semantic_cache_ttl: int = Field(default=3600, alias="SEMANTIC_CACHE_TTL")

    # Knowledge Base & Prompt Versioning
    knowledge_base_version: str = Field(default="v1", alias="KNOWLEDGE_BASE_VERSION")
    prompt_version: str = Field(default="v1", alias="PROMPT_VERSION")

    # Local Development Mode
    local_dev_mode: bool = Field(default=True, alias="LOCAL_DEV_MODE")

    # Paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)

    @property
    def qdrant_collection_name(self) -> str:
        return f"legal_articles_{self.knowledge_base_version}"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }


settings = Settings()
