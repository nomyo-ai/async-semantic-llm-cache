"""Redis distributed storage backend (async via redis.asyncio)."""

import json
from typing import Any, Optional

try:
    from redis import asyncio as aioredis
except ImportError as err:
    raise ImportError(
        "Redis backend requires 'redis' package. "
        "Install with: pip install semantic-llm-cache[redis]"
    ) from err

from semantic_llm_cache.backends.base import BaseBackend
from semantic_llm_cache.config import CacheEntry
from semantic_llm_cache.exceptions import CacheBackendError


class RedisBackend(BaseBackend):
    """Redis-based distributed cache storage (async).

    Uses redis.asyncio (bundled with redis>=4.2) for non-blocking I/O.
    The connection is created in __init__; no explicit connect() call needed
    as redis.asyncio uses a connection pool that connects lazily.
    """

    DEFAULT_PREFIX = "semantic_llm_cache:"

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        prefix: str = DEFAULT_PREFIX,
        **kwargs: Any,
    ) -> None:
        """Initialize Redis backend.

        Args:
            url: Redis connection URL
            prefix: Key prefix for cache entries
            **kwargs: Additional arguments passed to redis.asyncio.from_url
        """
        super().__init__()
        self._prefix = prefix.rstrip(":") + ":"
        self._redis = aioredis.from_url(url, **kwargs)

    async def ping(self) -> None:
        """Test Redis connection. Call this after construction to verify connectivity.

        Raises:
            CacheBackendError: If Redis is not reachable
        """
        try:
            await self._redis.ping()
        except Exception as e:
            raise CacheBackendError(f"Failed to connect to Redis: {e}") from e

    def _make_key(self, key: str) -> str:
        """Create full Redis key with prefix."""
        return f"{self._prefix}{key}"

    def _entry_to_dict(self, entry: CacheEntry) -> dict[str, Any]:
        """Convert CacheEntry to dictionary for storage."""
        return {
            "prompt": entry.prompt,
            "response": entry.response,
            "embedding": entry.embedding,
            "created_at": entry.created_at,
            "ttl": entry.ttl,
            "namespace": entry.namespace,
            "hit_count": entry.hit_count,
            "input_tokens": entry.input_tokens,
            "output_tokens": entry.output_tokens,
        }

    def _dict_to_entry(self, data: dict[str, Any]) -> CacheEntry:
        """Convert dictionary from storage to CacheEntry."""
        return CacheEntry(
            prompt=data["prompt"],
            response=data["response"],
            embedding=data.get("embedding"),
            created_at=data["created_at"],
            ttl=data.get("ttl"),
            namespace=data.get("namespace", "default"),
            hit_count=data.get("hit_count", 0),
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
        )

    async def get(self, key: str) -> Optional[CacheEntry]:
        """Retrieve cache entry by key.

        Args:
            key: Cache key to retrieve

        Returns:
            CacheEntry if found and not expired, None otherwise
        """
        try:
            redis_key = self._make_key(key)
            data = await self._redis.get(redis_key)

            if data is None:
                self._increment_misses()
                return None

            entry_dict = json.loads(data)
            entry = self._dict_to_entry(entry_dict)

            if self._check_expired(entry):
                await self.delete(key)
                self._increment_misses()
                return None

            self._increment_hits()
            entry.hit_count += 1

            entry_dict["hit_count"] = entry.hit_count
            await self._redis.set(redis_key, json.dumps(entry_dict))

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
            redis_key = self._make_key(key)
            data = json.dumps(self._entry_to_dict(entry))
            redis_ttl = entry.ttl if entry.ttl is not None else 0
            await self._redis.set(redis_key, data, ex=redis_ttl if redis_ttl > 0 else None)
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
            result = await self._redis.delete(self._make_key(key))
            return result > 0
        except Exception as e:
            raise CacheBackendError(f"Failed to delete entry: {e}") from e

    async def clear(self) -> None:
        """Clear all cache entries with this prefix."""
        try:
            keys = await self._redis.keys(f"{self._prefix}*")
            if keys:
                await self._redis.delete(*keys)
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
            keys = await self._redis.keys(f"{self._prefix}*")
            results = []

            for full_key in keys:
                short_key = full_key.decode().replace(self._prefix, "", 1)
                data = await self._redis.get(full_key)

                if data:
                    entry_dict = json.loads(data)
                    entry = self._dict_to_entry(entry_dict)

                    if namespace is None or entry.namespace == namespace:
                        if not self._check_expired(entry):
                            results.append((short_key, entry))

            return results
        except Exception as e:
            raise CacheBackendError(f"Failed to iterate entries: {e}") from e

    async def find_similar(
        self,
        embedding: list[float],
        threshold: float,
        namespace: Optional[str] = None,
    ) -> Optional[tuple[str, CacheEntry, float]]:
        """Find semantically similar cached entry.

        Note: Loads all entries for cosine scan. For large datasets consider
        Redis Stack with vector search (RediSearch).

        Args:
            embedding: Query embedding vector
            threshold: Minimum similarity score (0-1)
            namespace: Optional namespace filter

        Returns:
            (key, entry, similarity) tuple if found above threshold, None otherwise
        """
        try:
            entries = await self.iterate(namespace)
            candidates = [(k, v) for k, v in entries if v.embedding is not None]
            return self._find_best_match(candidates, embedding, threshold)
        except Exception as e:
            raise CacheBackendError(f"Failed to find similar entry: {e}") from e

    async def get_stats(self) -> dict[str, Any]:
        """Get backend statistics."""
        base_stats = await super().get_stats()

        try:
            keys = await self._redis.keys(f"{self._prefix}*")
            return {
                **base_stats,
                "size": len(keys) if keys else 0,
                "prefix": self._prefix,
            }
        except Exception as e:
            return {**base_stats, "size": 0, "prefix": self._prefix, "error": str(e)}

    async def close(self) -> None:
        """Close Redis connection."""
        try:
            await self._redis.aclose()
        except Exception:
            pass
