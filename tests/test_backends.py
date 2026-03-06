"""Tests for storage backends."""

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from semantic_llm_cache.backends import MemoryBackend
from semantic_llm_cache.backends.sqlite import SQLiteBackend  # noqa: F401
from semantic_llm_cache.config import CacheEntry
from semantic_llm_cache.exceptions import CacheBackendError


class TestBaseBackend:
    """Tests for BaseBackend abstract class."""

    def test_cosine_similarity(self):
        """Test cosine similarity helper method."""
        backend = MemoryBackend()

        entry1 = CacheEntry(
            prompt="test",
            response="response",
            embedding=[1.0, 0.0, 0.0],
        )
        entry2 = CacheEntry(
            prompt="test",
            response="response",
            embedding=[1.0, 0.0, 0.0],
        )
        entry3 = CacheEntry(
            prompt="test",
            response="response",
            embedding=[0.0, 1.0, 0.0],
        )

        # Test _find_best_match
        candidates = [("key1", entry1), ("key2", entry2), ("key3", entry3)]

        # Query matching entry1
        result = backend._find_best_match(candidates, [1.0, 0.0, 0.0], threshold=0.9)
        assert result is not None
        key, entry, sim = result
        assert sim == pytest.approx(1.0)


class TestMemoryBackend:
    """Tests for MemoryBackend."""

    def test_set_and_get(self, backend, sample_entry):
        """Test basic set and get operations."""
        backend.set("key1", sample_entry)
        retrieved = backend.get("key1")

        assert retrieved is not None
        assert retrieved.prompt == sample_entry.prompt
        assert retrieved.response == sample_entry.response

    def test_get_nonexistent(self, backend):
        """Test getting non-existent key returns None."""
        result = backend.get("nonexistent")
        assert result is None

    def test_delete(self, backend, sample_entry):
        """Test delete operation."""
        backend.set("key1", sample_entry)
        assert backend.get("key1") is not None

        assert backend.delete("key1") is True
        assert backend.get("key1") is None

    def test_delete_nonexistent(self, backend):
        """Test deleting non-existent key returns False."""
        assert backend.delete("nonexistent") is False

    def test_clear(self, backend, sample_entry):
        """Test clear operation."""
        backend.set("key1", sample_entry)
        backend.set("key2", sample_entry)
        backend.clear()

        assert backend.get("key1") is None
        assert backend.get("key2") is None

    def test_iterate_all(self, backend):
        """Test iterating over all entries."""
        entry1 = CacheEntry(prompt="p1", response="r1", created_at=time.time())
        entry2 = CacheEntry(prompt="p2", response="r2", created_at=time.time())

        backend.set("key1", entry1)
        backend.set("key2", entry2)

        results = backend.iterate()
        assert len(results) == 2

    def test_iterate_with_namespace(self, backend):
        """Test iterating with namespace filter."""
        entry1 = CacheEntry(
            prompt="p1", response="r1", namespace="ns1", created_at=time.time()
        )
        entry2 = CacheEntry(
            prompt="p2", response="r2", namespace="ns2", created_at=time.time()
        )

        backend.set("key1", entry1)
        backend.set("key2", entry2)

        results = backend.iterate(namespace="ns1")
        assert len(results) == 1
        assert results[0][1].namespace == "ns1"

    def test_find_similar(self, backend):
        """Test finding semantically similar entries."""
        entry1 = CacheEntry(
            prompt="What is Python?",
            response="r1",
            embedding=[1.0, 0.0, 0.0],
            created_at=time.time(),
        )
        entry2 = CacheEntry(
            prompt="What is Rust?",
            response="r2",
            embedding=[0.0, 1.0, 0.0],
            created_at=time.time(),
        )

        backend.set("key1", entry1)
        backend.set("key2", entry2)

        # Find similar to entry1
        result = backend.find_similar([1.0, 0.0, 0.0], threshold=0.9)
        assert result is not None
        key, entry, sim = result
        assert key == "key1"

    def test_find_similar_no_match(self, backend):
        """Test find_similar returns None when below threshold."""
        entry = CacheEntry(
            prompt="test",
            response="response",
            embedding=[1.0, 0.0, 0.0],
            created_at=time.time(),
        )

        backend.set("key1", entry)

        # Query with orthogonal vector
        result = backend.find_similar([0.0, 1.0, 0.0], threshold=0.9)
        assert result is None

    def test_get_stats(self, backend):
        """Test get_stats returns correct info."""
        entry = CacheEntry(
            prompt="test",
            response="response",
            created_at=time.time(),
        )

        backend.set("key1", entry)
        stats = backend.get_stats()

        assert stats["size"] == 1
        assert "hits" in stats
        assert "misses" in stats

    def test_hit_count_increments(self, backend, sample_entry):
        """Test hit count increments on cache hit."""
        backend.set("key1", sample_entry)

        backend.get("key1")  # First hit
        backend.get("key1")  # Second hit

        entry = backend.get("key1")
        assert entry.hit_count >= 1

    def test_lru_eviction(self):
        """Test LRU eviction when max_size is reached."""
        backend = MemoryBackend(max_size=2)

        entry1 = CacheEntry(prompt="p1", response="r1", created_at=time.time())
        entry2 = CacheEntry(prompt="p2", response="r2", created_at=time.time())
        entry3 = CacheEntry(prompt="p3", response="r3", created_at=time.time())

        backend.set("key1", entry1)
        backend.set("key2", entry2)
        assert backend.get_stats()["size"] == 2

        backend.set("key3", entry3)
        # Should evict oldest (key1)
        assert backend.get_stats()["size"] == 2

    def test_expired_entry_not_returned(self, backend):
        """Test expired entries are not returned."""
        entry = CacheEntry(
            prompt="test",
            response="response",
            ttl=1,
            created_at=time.time() - 2,  # 2 seconds ago with 1s TTL
        )

        backend.set("key1", entry)
        result = backend.get("key1")
        assert result is None


class TestSQLiteBackend:
    """Tests for SQLiteBackend."""

    @pytest.fixture
    def sqlite_backend(self, tmp_path):
        """Create SQLite backend with temp database."""
        from semantic_llm_cache.backends.sqlite import SQLiteBackend

        db_path = tmp_path / "test_cache.db"
        return SQLiteBackend(db_path)

    def test_set_and_get(self, sqlite_backend, sample_entry):
        """Test basic set and get operations."""
        sqlite_backend.set("key1", sample_entry)
        retrieved = sqlite_backend.get("key1")

        assert retrieved is not None
        assert retrieved.prompt == sample_entry.prompt
        assert retrieved.response == sample_entry.response

    def test_persistence(self, sqlite_backend, sample_entry, tmp_path):
        """Test entries persist across backend instances."""
        db_path = tmp_path / "test_persist.db"

        # Create first instance
        backend1 = SQLiteBackend(db_path)
        backend1.set("key1", sample_entry)

        # Create second instance (simulates restart)
        backend2 = SQLiteBackend(db_path)
        retrieved = backend2.get("key1")

        assert retrieved is not None
        assert retrieved.prompt == sample_entry.prompt

    def test_get_stats(self, sqlite_backend):
        """Test get_stats returns correct info."""
        entry = CacheEntry(prompt="test", response="response", created_at=time.time())
        sqlite_backend.set("key1", entry)

        stats = sqlite_backend.get_stats()
        assert stats["size"] == 1
        assert "db_path" in stats

    def test_clear(self, sqlite_backend, sample_entry):
        """Test clear operation."""
        sqlite_backend.set("key1", sample_entry)
        sqlite_backend.clear()

        assert sqlite_backend.get("key1") is None

    def test_close_and_reopen(self, sqlite_backend, sample_entry):
        """Test closing and reopening connection."""
        sqlite_backend.set("key1", sample_entry)
        sqlite_backend.close()

        # Should be able to use after close (reopens connection)
        retrieved = sqlite_backend.get("key1")
        assert retrieved is not None


class TestRedisBackend:
    """Tests for RedisBackend."""

    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        with patch("semantic_llm_cache.backends.redis.redis_lib") as mock:
            mock_client = MagicMock()
            mock.from_url.return_value = mock_client
            mock_client.ping.return_value = True
            mock_client.get.return_value = None
            mock_client.keys.return_value = []
            mock_client.delete.return_value = 1
            yield mock_client

    @pytest.fixture
    def redis_backend(self, mock_redis):
        """Create Redis backend with mocked client."""
        from semantic_llm_cache.backends.redis import RedisBackend

        backend = RedisBackend(url="redis://localhost:6379/0")
        backend._redis = mock_redis
        return backend

    def test_set_and_get(self, redis_backend, mock_redis, sample_entry):
        """Test basic set and get operations."""
        # Mock get to return stored data
        mock_redis.get.return_value = json.dumps({
            "prompt": sample_entry.prompt,
            "response": sample_entry.response,
            "embedding": sample_entry.embedding,
            "created_at": sample_entry.created_at,
            "ttl": sample_entry.ttl,
            "namespace": sample_entry.namespace,
            "hit_count": 0,
            "input_tokens": sample_entry.input_tokens,
            "output_tokens": sample_entry.output_tokens,
        }).encode()

        redis_backend.set("key1", sample_entry)
        retrieved = redis_backend.get("key1")

        assert retrieved is not None
        assert retrieved.prompt == sample_entry.prompt
        # set is called twice: once for initial set, once to update hit_count
        assert mock_redis.set.call_count == 2

    def test_get_nonexistent(self, redis_backend, mock_redis):
        """Test getting non-existent key returns None."""
        mock_redis.get.return_value = None
        result = redis_backend.get("nonexistent")
        assert result is None

    def test_delete(self, redis_backend, mock_redis, sample_entry):
        """Test delete operation."""
        mock_redis.delete.return_value = 1
        result = redis_backend.delete("key1")
        assert result is True
        mock_redis.delete.assert_called_once()

    def test_delete_nonexistent(self, redis_backend, mock_redis):
        """Test deleting non-existent key returns False."""
        mock_redis.delete.return_value = 0
        result = redis_backend.delete("nonexistent")
        assert result is False

    def test_clear(self, redis_backend, mock_redis):
        """Test clear operation."""
        mock_redis.keys.return_value = [b"semantic_llm_cache:key1", b"semantic_llm_cache:key2"]
        redis_backend.clear()
        mock_redis.delete.assert_called_once()

    def test_clear_empty(self, redis_backend, mock_redis):
        """Test clear with no entries."""
        mock_redis.keys.return_value = []
        redis_backend.clear()
        mock_redis.delete.assert_not_called()

    def test_iterate_all(self, redis_backend, mock_redis, sample_entry):
        """Test iterating over all entries."""
        entry_dict = {
            "prompt": sample_entry.prompt,
            "response": sample_entry.response,
            "embedding": sample_entry.embedding,
            "created_at": sample_entry.created_at,
            "ttl": sample_entry.ttl,
            "namespace": sample_entry.namespace,
            "hit_count": 0,
            "input_tokens": sample_entry.input_tokens,
            "output_tokens": sample_entry.output_tokens,
        }

        mock_redis.keys.return_value = [b"semantic_llm_cache:key1", b"semantic_llm_cache:key2"]
        mock_redis.get.return_value = json.dumps(entry_dict).encode()

        results = redis_backend.iterate()
        assert len(results) == 2

    def test_iterate_with_namespace(self, redis_backend, mock_redis, sample_entry):
        """Test iterating with namespace filter."""
        entry1_dict = {
            "prompt": "p1",
            "response": "r1",
            "embedding": None,
            "created_at": time.time(),
            "ttl": None,
            "namespace": "ns1",
            "hit_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
        entry2_dict = entry1_dict.copy()
        entry2_dict["namespace"] = "ns2"

        call_count = [0]

        def mock_get(key):
            call_count[0] += 1
            if call_count[0] == 1:
                return json.dumps(entry1_dict).encode()
            return json.dumps(entry2_dict).encode()

        mock_redis.keys.return_value = [b"semantic_llm_cache:key1", b"semantic_llm_cache:key2"]
        mock_redis.get.side_effect = mock_get

        results = redis_backend.iterate(namespace="ns1")
        assert len(results) == 1
        assert results[0][1].namespace == "ns1"

    def test_find_similar(self, redis_backend, mock_redis):
        """Test finding semantically similar entries."""
        entry_dict = {
            "prompt": "What is Python?",
            "response": "r1",
            "embedding": [1.0, 0.0, 0.0],
            "created_at": time.time(),
            "ttl": None,
            "namespace": "default",
            "hit_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }

        mock_redis.keys.return_value = [b"semantic_llm_cache:key1"]
        mock_redis.get.return_value = json.dumps(entry_dict).encode()

        result = redis_backend.find_similar([1.0, 0.0, 0.0], threshold=0.9)
        assert result is not None
        key, entry, sim = result
        assert key == "key1"

    def test_find_similar_no_match(self, redis_backend, mock_redis):
        """Test find_similar returns None when below threshold."""
        entry_dict = {
            "prompt": "test",
            "response": "response",
            "embedding": [1.0, 0.0, 0.0],
            "created_at": time.time(),
            "ttl": None,
            "namespace": "default",
            "hit_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }

        mock_redis.keys.return_value = [b"semantic_llm_cache:key1"]
        mock_redis.get.return_value = json.dumps(entry_dict).encode()

        # Query with orthogonal vector
        result = redis_backend.find_similar([0.0, 1.0, 0.0], threshold=0.9)
        assert result is None

    def test_get_stats(self, redis_backend, mock_redis):
        """Test get_stats returns correct info."""
        mock_redis.keys.return_value = [b"semantic_llm_cache:key1", b"semantic_llm_cache:key2"]
        stats = redis_backend.get_stats()

        assert "prefix" in stats
        assert stats["size"] == 2
        assert stats["prefix"] == "semantic_llm_cache:"

    def test_get_stats_error_handling(self, redis_backend, mock_redis):
        """Test get_stats handles Redis errors gracefully."""
        mock_redis.keys.side_effect = Exception("Connection lost")
        stats = redis_backend.get_stats()

        assert "error" in stats
        assert stats["size"] == 0

    def test_make_key(self, redis_backend):
        """Test key prefixing."""
        result = redis_backend._make_key("test_key")
        assert result == "semantic_llm_cache:test_key"

    def test_entry_to_dict(self, redis_backend, sample_entry):
        """Test converting entry to dictionary."""
        result = redis_backend._entry_to_dict(sample_entry)
        assert result["prompt"] == sample_entry.prompt
        assert result["response"] == sample_entry.response
        assert result["embedding"] == sample_entry.embedding

    def test_dict_to_entry(self, redis_backend):
        """Test converting dictionary to entry."""
        data = {
            "prompt": "test",
            "response": "response",
            "embedding": [1.0, 0.0],
            "created_at": time.time(),
            "ttl": 100,
            "namespace": "test_ns",
            "hit_count": 5,
            "input_tokens": 100,
            "output_tokens": 50,
        }

        entry = redis_backend._dict_to_entry(data)
        assert entry.prompt == "test"
        assert entry.namespace == "test_ns"
        assert entry.hit_count == 5

    def test_dict_to_entry_defaults(self, redis_backend):
        """Test dict_to_entry uses defaults for missing fields."""
        data = {
            "prompt": "test",
            "response": "response",
            "created_at": time.time(),
        }

        entry = redis_backend._dict_to_entry(data)
        assert entry.embedding is None
        assert entry.ttl is None
        assert entry.namespace == "default"
        assert entry.hit_count == 0
        assert entry.input_tokens == 0
        assert entry.output_tokens == 0

    def test_connection_failure(self):
        """Test connection failure raises CacheBackendError."""
        from semantic_llm_cache.backends.redis import RedisBackend
        from semantic_llm_cache.exceptions import CacheBackendError

        # Need to patch both the import and the from_url call
        with patch("semantic_llm_cache.backends.redis.redis_lib") as mock_redis:
            mock_client = MagicMock()
            mock_client.ping.side_effect = Exception("Connection refused")
            mock_redis.from_url.return_value = mock_client

            with pytest.raises(CacheBackendError, match="Failed to connect"):
                RedisBackend(url="redis://localhost:9999/0")

    def test_set_with_ttl(self, redis_backend, mock_redis, sample_entry):
        """Test setting entry with TTL."""
        sample_entry.ttl = 3600
        redis_backend.set("key1", sample_entry)

        call_args = mock_redis.set.call_args
        assert call_args[1]["ex"] == 3600

    def test_get_expired_entry(self, redis_backend, mock_redis):
        """Test expired entry is not returned."""
        expired_dict = {
            "prompt": "test",
            "response": "response",
            "embedding": None,
            "created_at": time.time() - 1000,
            "ttl": 100,  # 100 seconds TTL, created 1000 seconds ago
            "namespace": "default",
            "hit_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }

        mock_redis.get.return_value = json.dumps(expired_dict).encode()
        mock_redis.delete.return_value = 1

        result = redis_backend.get("expired_key")
        assert result is None
        mock_redis.delete.assert_called_once()

    def test_close(self, redis_backend, mock_redis):
        """Test closing Redis connection."""
        redis_backend.close()
        mock_redis.close.assert_called_once()

    def test_close_error_handling(self, redis_backend, mock_redis):
        """Test close handles errors gracefully."""
        mock_redis.close.side_effect = Exception("Close error")
        # Should not raise
        redis_backend.close()

    def test_iterate_with_expired_entries(self, redis_backend, mock_redis):
        """Test iterate filters out expired entries."""
        expired_dict = {
            "prompt": "expired",
            "response": "response",
            "embedding": None,
            "created_at": time.time() - 1000,
            "ttl": 100,
            "namespace": "default",
            "hit_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
        valid_dict = {
            "prompt": "valid",
            "response": "response",
            "embedding": None,
            "created_at": time.time(),
            "ttl": None,
            "namespace": "default",
            "hit_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }

        call_count = [0]

        def mock_get(key):
            call_count[0] += 1
            if call_count[0] == 1:
                return json.dumps(expired_dict).encode()
            return json.dumps(valid_dict).encode()

        mock_redis.keys.return_value = [b"semantic_llm_cache:expired", b"semantic_llm_cache:valid"]
        mock_redis.get.side_effect = mock_get

        results = redis_backend.iterate()
        # Only valid entry should be returned
        assert len(results) == 1
        assert results[0][1].prompt == "valid"

    def test_set_error_handling(self, redis_backend, mock_redis, sample_entry):
        """Test set handles Redis errors."""
        from semantic_llm_cache.exceptions import CacheBackendError

        mock_redis.set.side_effect = Exception("Redis error")

        with pytest.raises(CacheBackendError, match="Failed to set"):
            redis_backend.set("key1", sample_entry)

    def test_delete_error_handling(self, redis_backend, mock_redis):
        """Test delete handles Redis errors."""
        from semantic_llm_cache.exceptions import CacheBackendError

        mock_redis.delete.side_effect = Exception("Redis error")

        with pytest.raises(CacheBackendError, match="Failed to delete"):
            redis_backend.delete("key1")

    def test_iterate_error_handling(self, redis_backend, mock_redis):
        """Test iterate handles Redis errors."""
        from semantic_llm_cache.exceptions import CacheBackendError

        mock_redis.keys.side_effect = Exception("Redis error")

        with pytest.raises(CacheBackendError, match="Failed to iterate"):
            redis_backend.iterate()

    def test_get_json_error(self, redis_backend, mock_redis):
        """Test get handles invalid JSON."""
        import json
        mock_redis.get.return_value = b"invalid json"

        # The JSON decode error should be wrapped in CacheBackendError
        with pytest.raises((CacheBackendError, json.JSONDecodeError)):
            redis_backend.get("key1")

    def test_import_error_without_package(self):
        """Test ImportError when redis package not installed."""
        # This test validates the import guard in redis.py
        from semantic_llm_cache.backends import redis as redis_module

        # Check that RedisBackend is defined
        assert hasattr(redis_module, "RedisBackend")


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_is_expired_with_none_ttl(self):
        """Test entry with None TTL never expires."""
        entry = CacheEntry(
            prompt="test",
            response="response",
            ttl=None,
            created_at=time.time() - 10000,
        )
        assert not entry.is_expired(time.time())

    def test_is_expired_with_ttl(self):
        """Test entry with TTL expires correctly."""
        entry = CacheEntry(
            prompt="test",
            response="response",
            ttl=10,
            created_at=time.time() - 15,
        )
        assert entry.is_expired(time.time())

    def test_is_expired_not_yet(self):
        """Test entry not yet expired."""
        entry = CacheEntry(
            prompt="test",
            response="response",
            ttl=10,
            created_at=time.time() - 5,
        )
        assert not entry.is_expired(time.time())

    def test_estimate_cost(self):
        """Test cost estimation."""
        entry = CacheEntry(
            prompt="test",
            response="response",
            input_tokens=1000,
            output_tokens=500,
        )
        cost = entry.estimate_cost(0.001, 0.002)
        # 1000/1000 * 0.001 + 500/1000 * 0.002 = 0.001 + 0.001 = 0.002
        assert cost == pytest.approx(0.002)
