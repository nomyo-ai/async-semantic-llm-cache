"""Base backend implementation with common functionality."""

import time
from typing import Any, Optional

import numpy as np

from semantic_llm_cache.config import CacheEntry
from semantic_llm_cache.storage import StorageBackend


def cosine_similarity(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    """Calculate cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Similarity score between 0 and 1
    """
    a_arr = np.asarray(a, dtype=np.float32)
    b_arr = np.asarray(b, dtype=np.float32)

    dot_product = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot_product / (norm_a * norm_b))


class BaseBackend(StorageBackend):
    """Base backend with common sync helpers; async public interface via StorageBackend."""

    def __init__(self) -> None:
        """Initialize base backend."""
        self._hits: int = 0
        self._misses: int = 0

    def _increment_hits(self) -> None:
        """Increment hit counter."""
        self._hits += 1

    def _increment_misses(self) -> None:
        """Increment miss counter."""
        self._misses += 1

    def _check_expired(self, entry: CacheEntry) -> bool:
        """Check if entry is expired.

        Args:
            entry: CacheEntry to check

        Returns:
            True if expired, False otherwise
        """
        return entry.is_expired(time.time())

    def _find_best_match(
        self,
        candidates: list[tuple[str, CacheEntry]],
        query_embedding: list[float],
        threshold: float,
    ) -> Optional[tuple[str, CacheEntry, float]]:
        """Find best matching entry from candidates.

        Sync helper — CPU-only numpy ops, safe to call from async context.

        Args:
            candidates: List of (key, entry) tuples
            query_embedding: Query embedding vector
            threshold: Minimum similarity threshold

        Returns:
            (key, entry, similarity) tuple if found above threshold, None otherwise
        """
        best_match: Optional[tuple[str, CacheEntry, float]] = None
        best_similarity = threshold

        for key, entry in candidates:
            if entry.embedding is None:
                continue

            similarity = cosine_similarity(query_embedding, entry.embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = (key, entry, similarity)

        return best_match

    async def get_stats(self) -> dict[str, Any]:
        """Get backend statistics.

        Returns:
            Dictionary with hits and misses
        """
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / max(self._hits + self._misses, 1),
        }
