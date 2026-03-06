"""In-memory storage backend."""

import sys
from typing import Any, Optional

from semantic_llm_cache.backends.base import BaseBackend
from semantic_llm_cache.config import CacheEntry
from semantic_llm_cache.exceptions import CacheBackendError


class MemoryBackend(BaseBackend):
    """In-memory cache storage with LRU eviction.

    All operations are in-memory dict access — no I/O — so async methods
    run directly in the event loop without thread offloading.
    """

    def __init__(self, max_size: Optional[int] = None) -> None:
        """Initialize memory backend.

        Args:
            max_size: Maximum number of entries to store (LRU eviction when reached)
        """
        super().__init__()
        self._cache: dict[str, CacheEntry] = {}
        self._access_order: dict[str, float] = {}
        self._max_size = max_size
        self._access_counter: float = 0.0

    def _evict_if_needed(self) -> None:
        """Evict oldest entry if at capacity."""
        if self._max_size is None or len(self._cache) < self._max_size:
            return

        if self._access_order:
            lru_key = min(self._access_order, key=lambda k: self._access_order.get(k, 0))
            del self._cache[lru_key]
            del self._access_order[lru_key]

    def _update_access_time(self, key: str) -> None:
        """Update access time for LRU tracking."""
        self._access_counter += 1
        self._access_order[key] = self._access_counter

    async def get(self, key: str) -> Optional[CacheEntry]:
        """Retrieve cache entry by key.

        Args:
            key: Cache key to retrieve

        Returns:
            CacheEntry if found and not expired, None otherwise
        """
        try:
            entry = self._cache.get(key)
            if entry is None:
                self._increment_misses()
                return None

            if self._check_expired(entry):
                await self.delete(key)
                self._increment_misses()
                return None

            self._increment_hits()
            self._update_access_time(key)
            entry.hit_count += 1
            return entry
        except Exception as e:
            raise CacheBackendError(f"Failed to get entry: {e}") from e

    async def set(self, key: str, entry: CacheEntry) -> None:
        """Store cache entry.

        Args:
            key: Cache key to store under
            entry: CacheEntry to store
        """
        try:
            self._evict_if_needed()
            self._cache[key] = entry
            self._update_access_time(key)
        except Exception as e:
            raise CacheBackendError(f"Failed to set entry: {e}") from e

    async def delete(self, key: str) -> bool:
        """Delete cache entry.

        Args:
            key: Cache key to delete

        Returns:
            True if entry was deleted, False if not found
        """
        try:
            if key in self._cache:
                del self._cache[key]
                self._access_order.pop(key, None)
                return True
            return False
        except Exception as e:
            raise CacheBackendError(f"Failed to delete entry: {e}") from e

    async def clear(self) -> None:
        """Clear all cache entries."""
        try:
            self._cache.clear()
            self._access_order.clear()
        except Exception as e:
            raise CacheBackendError(f"Failed to clear cache: {e}") from e

    async def iterate(
        self, namespace: Optional[str] = None
    ) -> list[tuple[str, CacheEntry]]:
        """Iterate over cache entries, optionally filtered by namespace.

        Args:
            namespace: Optional namespace filter

        Returns:
            List of (key, entry) tuples
        """
        try:
            if namespace is None:
                return list(self._cache.items())

            return [
                (k, v)
                for k, v in self._cache.items()
                if v.namespace == namespace and not self._check_expired(v)
            ]
        except Exception as e:
            raise CacheBackendError(f"Failed to iterate entries: {e}") from e

    async def find_similar(
        self,
        embedding: list[float],
        threshold: float,
        namespace: Optional[str] = None,
    ) -> Optional[tuple[str, CacheEntry, float]]:
        """Find semantically similar cached entry.

        Args:
            embedding: Query embedding vector
            threshold: Minimum similarity score (0-1)
            namespace: Optional namespace filter

        Returns:
            (key, entry, similarity) tuple if found above threshold, None otherwise
        """
        try:
            candidates = [
                (k, v)
                for k, v in self._cache.items()
                if v.embedding is not None
                and not self._check_expired(v)
                and (namespace is None or v.namespace == namespace)
            ]
            return self._find_best_match(candidates, embedding, threshold)
        except Exception as e:
            raise CacheBackendError(f"Failed to find similar entry: {e}") from e

    async def get_stats(self) -> dict[str, Any]:
        """Get backend statistics.

        Returns:
            Dictionary with size, memory usage, hits, misses
        """
        base_stats = await super().get_stats()
        memory_usage = sys.getsizeof(self._cache) + sum(
            sys.getsizeof(k) + sys.getsizeof(v) for k, v in self._cache.items()
        )

        return {
            **base_stats,
            "size": len(self._cache),
            "memory_bytes": memory_usage,
            "max_size": self._max_size,
        }
