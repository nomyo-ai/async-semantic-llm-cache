"""Statistics and analytics for llm-semantic-cache."""

from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable, Optional

from semantic_llm_cache.backends import MemoryBackend
from semantic_llm_cache.backends.base import BaseBackend


@dataclass
class CacheStats:
    """Statistics for cache performance."""

    hits: int = 0
    misses: int = 0
    total_saved_ms: float = 0.0
    estimated_savings_usd: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / max(total, 1)

    @property
    def total_requests(self) -> int:
        """Get total number of requests."""
        return self.hits + self.misses

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "total_requests": self.total_requests,
            "total_saved_ms": self.total_saved_ms,
            "estimated_savings_usd": self.estimated_savings_usd,
        }

    def __iadd__(self, other: "CacheStats") -> "CacheStats":
        self.hits += other.hits
        self.misses += other.misses
        self.total_saved_ms += other.total_saved_ms
        self.estimated_savings_usd += other.estimated_savings_usd
        return self


class _StatsManager:
    """Manager for global cache statistics.

    Uses threading.Lock for record_hit/record_miss — these are simple counter
    increments with no awaits inside the lock, so threading.Lock is safe and
    avoids the overhead of asyncio.Lock for hot-path calls.
    """

    def __init__(self) -> None:
        """Initialize stats manager."""
        self._stats: dict[str, CacheStats] = {}
        self._lock = Lock()
        self._default_backend: Optional[BaseBackend] = None

    def get_backend(self) -> BaseBackend:
        """Get default backend for cache operations."""
        if self._default_backend is None:
            self._default_backend = MemoryBackend()
        return self._default_backend

    def set_backend(self, backend: BaseBackend) -> None:
        """Set default backend for cache operations."""
        with self._lock:
            self._default_backend = backend

    def record_hit(
        self,
        namespace: str,
        latency_saved_ms: float = 0.0,
        saved_cost: float = 0.0,
    ) -> None:
        """Record a cache hit (sync, safe to call from async context)."""
        with self._lock:
            if namespace not in self._stats:
                self._stats[namespace] = CacheStats()
            stats = self._stats[namespace]
            stats.hits += 1
            stats.total_saved_ms += latency_saved_ms
            stats.estimated_savings_usd += saved_cost

    def record_miss(self, namespace: str) -> None:
        """Record a cache miss (sync, safe to call from async context)."""
        with self._lock:
            if namespace not in self._stats:
                self._stats[namespace] = CacheStats()
            self._stats[namespace].misses += 1

    def get_stats(self, namespace: Optional[str] = None) -> CacheStats:
        """Get statistics for namespace or all."""
        with self._lock:
            if namespace is not None:
                return self._stats.get(namespace, CacheStats())

            total = CacheStats()
            for stats in self._stats.values():
                total += stats
            return total

    def clear_stats(self, namespace: Optional[str] = None) -> None:
        """Clear statistics for namespace or all."""
        with self._lock:
            if namespace is None:
                self._stats.clear()
            elif namespace in self._stats:
                del self._stats[namespace]


# Global stats manager instance
_stats_manager = _StatsManager()


def get_stats(namespace: Optional[str] = None) -> dict[str, Any]:
    """Get cache statistics (sync).

    Args:
        namespace: Optional namespace to filter by

    Returns:
        Dictionary with cache statistics
    """
    return _stats_manager.get_stats(namespace).to_dict()


async def clear_cache(namespace: Optional[str] = None) -> int:
    """Clear all cached entries (async).

    Args:
        namespace: Optional namespace to clear (None = all)

    Returns:
        Number of entries cleared
    """
    backend = _stats_manager.get_backend()

    if namespace is None:
        stats = await backend.get_stats()
        size = stats.get("size", 0)
        await backend.clear()
        _stats_manager.clear_stats()
        return size

    entries = await backend.iterate(namespace=namespace)
    count = len(entries)
    for key, _ in entries:
        await backend.delete(key)
    _stats_manager.clear_stats(namespace)
    return count


async def invalidate(
    pattern: str,
    namespace: Optional[str] = None,
) -> int:
    """Invalidate cache entries matching pattern (async).

    Args:
        pattern: String pattern to match in prompts
        namespace: Optional namespace filter

    Returns:
        Number of entries invalidated
    """
    backend = _stats_manager.get_backend()
    entries = await backend.iterate(namespace=namespace)
    count = 0
    pattern_lower = pattern.lower()

    for key, entry in entries:
        if pattern_lower in entry.prompt.lower():
            await backend.delete(key)
            count += 1

    return count


async def warm_cache(
    prompts: list[str],
    llm_func: Callable[[str], Any],
    namespace: str = "default",
) -> int:
    """Pre-populate cache with prompts (async).

    Args:
        prompts: List of prompts to cache
        llm_func: Async or sync LLM function to call for each prompt
        namespace: Cache namespace to use

    Returns:
        Number of prompts attempted
    """
    import asyncio
    import inspect

    from semantic_llm_cache.core import cache

    cached_func = cache(namespace=namespace)(llm_func)

    for prompt in prompts:
        try:
            result = cached_func(prompt)
            if inspect.isawaitable(result):
                await result
        except Exception:
            pass

    return len(prompts)


async def export_cache(
    namespace: Optional[str] = None,
    filepath: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Export cache entries for analysis (async).

    Args:
        namespace: Optional namespace filter
        filepath: Optional file path to save export (JSON)

    Returns:
        List of cache entry dictionaries
    """
    import json
    from datetime import datetime

    backend = _stats_manager.get_backend()
    entries = await backend.iterate(namespace=namespace)

    export_data = []
    for key, entry in entries:
        export_data.append({
            "key": key,
            "prompt": entry.prompt,
            "response": str(entry.response)[:1000],
            "namespace": entry.namespace,
            "hit_count": entry.hit_count,
            "created_at": datetime.fromtimestamp(entry.created_at).isoformat(),
            "ttl": entry.ttl,
            "input_tokens": entry.input_tokens,
            "output_tokens": entry.output_tokens,
        })

    if filepath:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)

    return export_data
