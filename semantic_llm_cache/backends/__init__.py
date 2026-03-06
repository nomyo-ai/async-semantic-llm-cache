"""Storage backends for llm-semantic-cache."""

from semantic_llm_cache.backends.base import BaseBackend
from semantic_llm_cache.backends.memory import MemoryBackend

try:
    from semantic_llm_cache.backends.sqlite import SQLiteBackend
except ImportError:
    SQLiteBackend = None  # type: ignore

try:
    from semantic_llm_cache.backends.redis import RedisBackend
except ImportError:
    RedisBackend = None  # type: ignore

__all__ = [
    "BaseBackend",
    "MemoryBackend",
    "SQLiteBackend",
    "RedisBackend",
]
