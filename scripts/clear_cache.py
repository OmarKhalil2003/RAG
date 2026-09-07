import os
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from legal_rag.config import settings
from legal_rag.cache.semantic_cache import RedisSemanticCache


def clear_all_caches():
    print("=" * 60)
    print("Clearing All Caches (Semantic Cache & Temporary Files)")
    print("=" * 60)

    # 1. Clear Redis / In-Memory Semantic Cache
    cache = RedisSemanticCache()
    cache.clear()
    print("[SemanticCache] Cache flushed successfully.")

    # 2. Clear Python __pycache__ folders
    cleared_pycache = 0
    for p in project_root.rglob("__pycache__"):
        try:
            shutil.rmtree(p, ignore_errors=True)
            cleared_pycache += 1
        except Exception:
            pass
    print(f"[Python] Cleared {cleared_pycache} __pycache__ directories.")

    # 3. Clear pytest cache
    pytest_cache = project_root / ".pytest_cache"
    if pytest_cache.exists():
        shutil.rmtree(pytest_cache, ignore_errors=True)
        print("[Pytest] Cleared .pytest_cache directory.")

    print("=" * 60)
    print("All caches cleared successfully.")
    print("=" * 60)


if __name__ == "__main__":
    clear_all_caches()
