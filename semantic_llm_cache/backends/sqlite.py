"""SQLite persistent storage backend (async via aiosqlite)."""

import json
from pathlib import Path
from typing import Any, Optional

try:
    import aiosqlite
except ImportError as err:
    raise ImportError(
        "SQLite backend requires 'aiosqlite' package. "
        "Install with: pip install semantic-llm-cache[sqlite]"
    ) from err

from semantic_llm_cache.backends.base import BaseBackend
from semantic_llm_cache.config import CacheEntry
from semantic_llm_cache.exceptions import CacheBackendError


class SQLiteBackend(BaseBackend):
    """SQLite-based persistent cache storage (async).

    Uses aiosqlite for non-blocking I/O. A single persistent connection
    is opened lazily on first use and reused for all subsequent operations.
    """

    def __init__(self, db_path: str | Path = "semantic_cache.db") -> None:
        """Initialize SQLite backend.

        Args:
            db_path: Path to SQLite database file, or ":memory:" for in-memory DB
        """
        super().__init__()
        self._db_path = str(db_path) if isinstance(db_path, Path) else db_path
        self._conn: Optional[aiosqlite.Connection] = None

    async def _get_conn(self) -> aiosqlite.Connection:
        """Get or create the persistent async connection."""
        if self._conn is None:
            self._conn = await aiosqlite.connect(self._db_path)
            self._conn.row_factory = aiosqlite.Row
            await self._initialize_schema()
        return self._conn

    async def _initialize_schema(self) -> None:
        """Initialize database schema."""
        conn = await self._get_conn()
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_entries (
                key TEXT PRIMARY KEY,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                embedding TEXT,
                created_at REAL NOT NULL,
                ttl INTEGER,
                namespace TEXT NOT NULL DEFAULT 'default',
                hit_count INTEGER DEFAULT 0,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0
            )
            """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_namespace
            ON cache_entries(namespace)
            """
        )
        await conn.commit()

    def _row_to_entry(self, row: aiosqlite.Row) -> CacheEntry:
        """Convert database row to CacheEntry."""
        embedding = None
        if row["embedding"]:
            embedding = json.loads(row["embedding"])

        return CacheEntry(
            prompt=row["prompt"],
            response=json.loads(row["response"]),
            embedding=embedding,
            created_at=row["created_at"],
            ttl=row["ttl"],
            namespace=row["namespace"],
            hit_count=row["hit_count"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
        )

    async def get(self, key: str) -> Optional[CacheEntry]:
        """Retrieve cache entry by key.

        Args:
            key: Cache key to retrieve

        Returns:
            CacheEntry if found and not expired, None otherwise
        """
        try:
            conn = await self._get_conn()
            async with conn.execute(
                "SELECT * FROM cache_entries WHERE key = ?", (key,)
            ) as cursor:
                row = await cursor.fetchone()

            if row is None:
                self._increment_misses()
                return None

            entry = self._row_to_entry(row)

            if self._check_expired(entry):
                await self.delete(key)
                self._increment_misses()
                return None

            self._increment_hits()
            entry.hit_count += 1

            await conn.execute(
                "UPDATE cache_entries SET hit_count = hit_count + 1 WHERE key = ?",
                (key,),
            )
            await conn.commit()

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
            conn = await self._get_conn()
            embedding_json = json.dumps(entry.embedding) if entry.embedding else None
            response_json = json.dumps(entry.response)

            await conn.execute(
                """
                INSERT OR REPLACE INTO cache_entries
                (key, prompt, response, embedding, created_at, ttl, namespace,
                 hit_count, input_tokens, output_tokens)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    entry.prompt,
                    response_json,
                    embedding_json,
                    entry.created_at,
                    entry.ttl,
                    entry.namespace,
                    entry.hit_count,
                    entry.input_tokens,
                    entry.output_tokens,
                ),
            )
            await conn.commit()
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
            conn = await self._get_conn()
            async with conn.execute(
                "DELETE FROM cache_entries WHERE key = ?", (key,)
            ) as cursor:
                rowcount = cursor.rowcount
            await conn.commit()
            return rowcount > 0
        except Exception as e:
            raise CacheBackendError(f"Failed to delete entry: {e}") from e

    async def clear(self) -> None:
        """Clear all cache entries."""
        try:
            conn = await self._get_conn()
            await conn.execute("DELETE FROM cache_entries")
            await conn.commit()
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
            conn = await self._get_conn()

            if namespace is None:
                query = "SELECT key, * FROM cache_entries"
                params: tuple[()] = ()
            else:
                query = "SELECT key, * FROM cache_entries WHERE namespace = ?"
                params = (namespace,)

            async with conn.execute(query, params) as cursor:
                rows = await cursor.fetchall()

            results = []
            for row in rows:
                key = row["key"]
                entry = self._row_to_entry(row)
                if not self._check_expired(entry):
                    results.append((key, entry))

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
        """Get backend statistics.

        Returns:
            Dictionary with size, database path, hits, misses
        """
        base_stats = await super().get_stats()

        try:
            conn = await self._get_conn()
            async with conn.execute("SELECT COUNT(*) FROM cache_entries") as cursor:
                row = await cursor.fetchone()
            size = row[0] if row else 0

            return {
                **base_stats,
                "size": size,
                "db_path": self._db_path,
            }
        except Exception as e:
            return {**base_stats, "size": 0, "db_path": self._db_path, "error": str(e)}

    async def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
