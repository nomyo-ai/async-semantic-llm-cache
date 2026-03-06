"""Storage backend interface for prompt-cache."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from semantic_llm_cache.config import CacheEntry


class StorageBackend(ABC):
    """Abstract base class for async cache storage backends."""

    @abstractmethod
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Retrieve cache entry by key.

        Args:
            key: Cache key to retrieve

        Returns:
            CacheEntry if found and not expired, None otherwise

        Raises:
            CacheBackendError: If backend operation fails
        """
        pass

    @abstractmethod
    async def set(self, key: str, entry: CacheEntry) -> None:
        """Store cache entry.

        Args:
            key: Cache key to store under
            entry: CacheEntry to store

        Raises:
            CacheBackendError: If backend operation fails
        """
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete cache entry.

        Args:
            key: Cache key to delete

        Returns:
            True if entry was deleted, False if not found

        Raises:
            CacheBackendError: If backend operation fails
        """
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all cache entries.

        Raises:
            CacheBackendError: If backend operation fails
        """
        pass

    @abstractmethod
    async def iterate(self, namespace: Optional[str] = None) -> list[tuple[str, CacheEntry]]:
        """Iterate over cache entries, optionally filtered by namespace.

        Args:
            namespace: Optional namespace filter

        Returns:
            List of (key, entry) tuples

        Raises:
            CacheBackendError: If backend operation fails
        """
        pass

    @abstractmethod
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

        Raises:
            CacheBackendError: If backend operation fails
        """
        pass

    @abstractmethod
    async def get_stats(self) -> dict[str, Any]:
        """Get backend statistics.

        Returns:
            Dictionary with stats like size, memory_usage, etc.

        Raises:
            CacheBackendError: If backend operation fails
        """
        pass
