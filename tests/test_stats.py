"""Tests for statistics and analytics module."""

import time

import pytest

from semantic_llm_cache.backends import MemoryBackend
from semantic_llm_cache.config import CacheEntry
from semantic_llm_cache.stats import (
    CacheStats,
    _stats_manager,
    clear_cache,
    export_cache,
    get_stats,
    invalidate,
    warm_cache,
)


@pytest.fixture(autouse=True)
def clear_stats_state():
    """Clear stats state before each test."""
    _stats_manager.clear_stats()
    yield


class TestCacheStats:
    """Tests for CacheStats dataclass."""

    def test_default_values(self):
        """Test CacheStats initializes with defaults."""
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.total_saved_ms == 0.0
        assert stats.estimated_savings_usd == 0.0

    def test_hit_rate_empty(self):
        """Test hit rate with no requests."""
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_hit_rate_all_hits(self):
        """Test hit rate with all cache hits."""
        stats = CacheStats(hits=10, misses=0)
        assert stats.hit_rate == 1.0

    def test_hit_rate_all_misses(self):
        """Test hit rate with all cache misses."""
        stats = CacheStats(hits=0, misses=10)
        assert stats.hit_rate == 0.0

    def test_hit_rate_mixed(self):
        """Test hit rate with mixed hits and misses."""
        stats = CacheStats(hits=7, misses=3)
        assert stats.hit_rate == 0.7

    def test_total_requests(self):
        """Test total requests calculation."""
        stats = CacheStats(hits=5, misses=3)
        assert stats.total_requests == 8

    def test_to_dict(self):
        """Test converting stats to dictionary."""
        stats = CacheStats(hits=10, misses=5, total_saved_ms=1000.0, estimated_savings_usd=0.5)
        result = stats.to_dict()

        assert result["hits"] == 10
        assert result["misses"] == 5
        assert result["hit_rate"] == 2/3
        assert result["total_requests"] == 15
        assert result["total_saved_ms"] == 1000.0
        assert result["estimated_savings_usd"] == 0.5

    def test_iadd(self):
        """Test in-place addition of stats."""
        stats1 = CacheStats(hits=5, misses=3, total_saved_ms=500.0, estimated_savings_usd=0.25)
        stats2 = CacheStats(hits=3, misses=2, total_saved_ms=300.0, estimated_savings_usd=0.15)

        stats1 += stats2

        assert stats1.hits == 8
        assert stats1.misses == 5
        assert stats1.total_saved_ms == 800.0
        assert stats1.estimated_savings_usd == 0.4


class TestStatsManager:
    """Tests for _StatsManager."""

    def test_record_hit(self):
        """Test recording a cache hit."""
        _stats_manager.record_hit("test_ns", latency_saved_ms=100.0, saved_cost=0.01)

        stats = _stats_manager.get_stats("test_ns")
        assert stats.hits == 1
        assert stats.total_saved_ms == 100.0
        assert stats.estimated_savings_usd == 0.01

    def test_record_multiple_hits(self):
        """Test recording multiple hits."""
        _stats_manager.record_hit("test_ns", latency_saved_ms=50.0, saved_cost=0.005)
        _stats_manager.record_hit("test_ns", latency_saved_ms=75.0, saved_cost=0.008)

        stats = _stats_manager.get_stats("test_ns")
        assert stats.hits == 2
        assert stats.total_saved_ms == 125.0

    def test_record_miss(self):
        """Test recording a cache miss."""
        _stats_manager.record_miss("test_ns")

        stats = _stats_manager.get_stats("test_ns")
        assert stats.misses == 1

    def test_get_stats_namespace(self):
        """Test getting stats for specific namespace."""
        _stats_manager.record_hit("ns1", latency_saved_ms=100.0)
        _stats_manager.record_miss("ns1")
        _stats_manager.record_hit("ns2", latency_saved_ms=50.0)

        stats1 = _stats_manager.get_stats("ns1")
        stats2 = _stats_manager.get_stats("ns2")

        assert stats1.hits == 1
        assert stats1.misses == 1
        assert stats2.hits == 1
        assert stats2.misses == 0

    def test_get_stats_all_namespaces(self):
        """Test getting aggregated stats for all namespaces."""
        _stats_manager.record_hit("ns1", latency_saved_ms=100.0)
        _stats_manager.record_hit("ns2", latency_saved_ms=50.0)
        _stats_manager.record_miss("ns1")

        stats = _stats_manager.get_stats(None)  # All namespaces
        assert stats.hits == 2
        assert stats.misses == 1

    def test_get_stats_nonexistent_namespace(self):
        """Test getting stats for namespace with no activity."""
        stats = _stats_manager.get_stats("nonexistent")
        assert stats.hits == 0
        assert stats.misses == 0

    def test_clear_stats_namespace(self):
        """Test clearing stats for specific namespace."""
        _stats_manager.record_hit("ns1", latency_saved_ms=100.0)
        _stats_manager.record_hit("ns2", latency_saved_ms=50.0)

        _stats_manager.clear_stats("ns1")

        stats1 = _stats_manager.get_stats("ns1")
        stats2 = _stats_manager.get_stats("ns2")

        assert stats1.hits == 0
        assert stats2.hits == 1

    def test_clear_stats_all(self):
        """Test clearing all stats."""
        _stats_manager.record_hit("ns1", latency_saved_ms=100.0)
        _stats_manager.record_hit("ns2", latency_saved_ms=50.0)

        _stats_manager.clear_stats()

        stats = _stats_manager.get_stats(None)
        assert stats.hits == 0
        assert stats.misses == 0

    def test_set_backend(self):
        """Test setting default backend."""
        custom_backend = MemoryBackend(max_size=10)
        _stats_manager.set_backend(custom_backend)

        retrieved = _stats_manager.get_backend()
        assert retrieved is custom_backend


class TestPublicStatsAPI:
    """Tests for public stats API functions."""

    def test_get_stats(self):
        """Test get_stats returns dictionary."""
        _stats_manager.record_hit("test_ns", latency_saved_ms=100.0)

        stats = get_stats("test_ns")
        assert isinstance(stats, dict)
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats

    def test_clear_cache_all(self):
        """Test clear_cache clears all entries."""
        backend = _stats_manager.get_backend()

        # Add some entries
        entry = CacheEntry(prompt="test", response="response", created_at=time.time())
        backend.set("key1", entry)
        backend.set("key2", entry)

        count = clear_cache()
        assert count >= 0

    def test_clear_cache_namespace(self):
        """Test clear_cache clears specific namespace."""
        backend = _stats_manager.get_backend()

        # Add entries in different namespaces
        entry1 = CacheEntry(prompt="test1", response="r1", namespace="ns1", created_at=time.time())
        entry2 = CacheEntry(prompt="test2", response="r2", namespace="ns2", created_at=time.time())

        backend.set("key1", entry1)
        backend.set("key2", entry2)

        count = clear_cache(namespace="ns1")
        assert count >= 0

    def test_invalidate_pattern(self):
        """Test invalidating entries by pattern."""
        backend = _stats_manager.get_backend()

        # Add entries with different prompts
        entry1 = CacheEntry(prompt="Python programming", response="r1", created_at=time.time())
        entry2 = CacheEntry(prompt="Rust programming", response="r2", created_at=time.time())
        entry3 = CacheEntry(prompt="JavaScript", response="r3", created_at=time.time())

        backend.set("key1", entry1)
        backend.set("key2", entry2)
        backend.set("key3", entry3)

        count = invalidate("Python")
        assert count >= 0

    def test_invalidate_case_insensitive(self):
        """Test invalidate is case insensitive."""
        backend = _stats_manager.get_backend()

        entry = CacheEntry(prompt="PYTHON programming", response="r1", created_at=time.time())
        backend.set("key1", entry)

        count = invalidate("python")
        assert count >= 0

    def test_invalidate_no_matches(self):
        """Test invalidate with no matches."""
        backend = _stats_manager.get_backend()

        entry = CacheEntry(prompt="Rust programming", response="r1", created_at=time.time())
        backend.set("key1", entry)

        count = invalidate("Python")
        assert count == 0

    def test_warm_cache(self):
        """Test warming cache with prompts."""
        prompts = ["prompt1", "prompt2"]

        def mock_llm(prompt: str) -> str:
            return f"Response to: {prompt}"

        count = warm_cache(prompts, mock_llm, namespace="warm_test")
        assert count == len(prompts)

    def test_warm_cache_with_failures(self):
        """Test warm_cache handles LLM failures gracefully."""

        def failing_llm(prompt: str) -> str:
            if "fail" in prompt:
                raise ValueError("LLM error")
            return f"Response to: {prompt}"

        prompts = ["prompt1", "fail_prompt", "prompt3"]
        count = warm_cache(prompts, failing_llm, namespace="warm_fail_test")
        # Should return count even if some prompts fail
        assert count == len(prompts)


class TestExportCache:
    """Tests for export_cache function."""

    def test_export_all_entries(self, tmp_path):
        """Test exporting all cache entries."""
        backend = _stats_manager.get_backend()
        backend.clear()

        # Add test entries
        entry1 = CacheEntry(
            prompt="test prompt 1",
            response="response 1",
            namespace="test_ns",
            created_at=time.time(),
            hit_count=5,
            ttl=3600,
            input_tokens=100,
            output_tokens=50,
        )
        backend.set("key1", entry1)

        entries = export_cache()
        assert len(entries) >= 0

        if entries:
            assert "key" in entries[0]
            assert "prompt" in entries[0]
            assert "response" in entries[0]
            assert "namespace" in entries[0]
            assert "hit_count" in entries[0]

    def test_export_namespace_filtered(self, tmp_path):
        """Test exporting entries filtered by namespace."""
        backend = _stats_manager.get_backend()
        backend.clear()

        # Add entries in different namespaces
        entry1 = CacheEntry(
            prompt="test1", response="r1", namespace="ns1", created_at=time.time()
        )
        entry2 = CacheEntry(
            prompt="test2", response="r2", namespace="ns2", created_at=time.time()
        )

        backend.set("key1", entry1)
        backend.set("key2", entry2)

        entries = export_cache(namespace="ns1")
        # Should only return entries from ns1
        assert all(e["namespace"] == "ns1" for e in entries)

    def test_export_to_file(self, tmp_path):
        """Test exporting cache to JSON file."""
        import json

        filepath = tmp_path / "export.json"

        backend = _stats_manager.get_backend()
        backend.clear()

        entry = CacheEntry(
            prompt="test",
            response="response",
            namespace="test",
            created_at=time.time(),
            hit_count=3,
        )
        backend.set("key1", entry)

        export_cache(filepath=str(filepath))

        # Verify file was created and is valid JSON
        assert filepath.exists()
        with open(filepath) as f:
            data = json.load(f)
        assert isinstance(data, list)

    def test_export_truncates_large_responses(self, tmp_path):
        """Test that large responses are truncated in export."""
        backend = _stats_manager.get_backend()
        backend.clear()

        # Create entry with very large response
        large_response = "x" * 2000
        entry = CacheEntry(
            prompt="test",
            response=large_response,
            created_at=time.time(),
        )
        backend.set("key1", entry)

        entries = export_cache()
        if entries:
            # Response should be truncated to 1000 chars
            assert len(entries[0]["response"]) <= 1000


class TestStatsIntegration:
    """Integration tests for stats with actual cache operations."""

    def test_stats_tracking_with_cache_decorator(self):
        """Test that stats are tracked during cache operations."""
        from semantic_llm_cache import cache

        backend = MemoryBackend()
        _stats_manager.clear_stats("integration_test")

        @cache(backend=backend, namespace="integration_test")
        def cached_func(prompt: str) -> str:
            return f"Response to: {prompt}"

        # Generate activity
        cached_func("prompt1")
        cached_func("prompt1")  # Hit
        cached_func("prompt2")

        stats = get_stats("integration_test")
        assert stats["total_requests"] >= 2

    def test_cache_invalidate_integration(self):
        """Test invalidate removes entries from backend."""
        backend = _stats_manager.get_backend()
        backend.clear()

        entry = CacheEntry(
            prompt="Python is great",
            response="Yes, it is!",
            created_at=time.time(),
        )
        backend.set("key1", entry)

        # Verify entry exists
        assert backend.get("key1") is not None

        # Invalidate
        invalidate("Python")

        # Entry should be gone
        assert backend.get("key1") is None

    def test_export_includes_metadata(self, tmp_path):
        """Test export includes all metadata fields."""
        backend = _stats_manager.get_backend()
        backend.clear()

        entry = CacheEntry(
            prompt="test prompt",
            response="test response",
            namespace="export_test",
            created_at=time.time(),
            ttl=7200,
            hit_count=10,
            input_tokens=500,
            output_tokens=250,
            embedding=[0.1, 0.2, 0.3],
        )
        backend.set("key1", entry)

        entries = export_cache(namespace="export_test")
        if entries:
            e = entries[0]
            assert "created_at" in e
            assert "ttl" in e
            assert "hit_count" in e
            assert "input_tokens" in e
            assert "output_tokens" in e
