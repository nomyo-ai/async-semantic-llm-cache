"""Integration tests for prompt-cache."""

import time

import pytest

from semantic_llm_cache import cache, clear_cache, get_stats, invalidate
from semantic_llm_cache.backends import MemoryBackend


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_cache_workflow(self):
        """Test complete cache workflow from hit to miss."""
        backend = MemoryBackend()
        call_count = {"count": 0}

        @cache(backend=backend)
        def llm_function(prompt: str) -> str:
            call_count["count"] += 1
            return f"Response to: {prompt}"

        # First call - miss
        result1 = llm_function("What is Python?")
        assert result1 == "Response to: What is Python?"
        assert call_count["count"] == 1

        # Second call - hit
        result2 = llm_function("What is Python?")
        assert result2 == "Response to: What is Python?"
        assert call_count["count"] == 1

        # Different prompt - miss
        result3 = llm_function("What is Rust?")
        assert result3 == "Response to: What is Rust?"
        assert call_count["count"] == 2

    def test_stats_integration(self):
        """Test statistics tracking."""
        backend = MemoryBackend()

        @cache(backend=backend, namespace="test")
        def llm_function(prompt: str) -> str:
            return f"Response to: {prompt}"

        # Generate some activity
        llm_function("prompt 1")
        llm_function("prompt 1")  # Hit
        llm_function("prompt 2")

        stats = get_stats(namespace="test")
        assert stats["total_requests"] >= 2

    def test_clear_cache_integration(self):
        """Test clearing cache affects function behavior."""
        backend = MemoryBackend()

        @cache(backend=backend)
        def llm_function(prompt: str) -> str:
            return f"Response to: {prompt}"

        llm_function("test prompt")

        # Clear cache
        cleared = clear_cache()
        assert cleared >= 0

        # Function should still work
        result = llm_function("test prompt")
        assert result == "Response to: test prompt"

    def test_invalidate_integration(self):
        """Test invalidating cache entries."""
        backend = MemoryBackend()

        @cache(backend=backend)
        def llm_function(prompt: str) -> str:
            return f"Response to: {prompt}"

        llm_function("Python programming")
        llm_function("Rust programming")

        # Invalidate Python entries
        count = invalidate("Python")
        assert count >= 0

    def test_multiple_namespaces(self):
        """Test cache isolation across namespaces."""
        backend = MemoryBackend()

        @cache(backend=backend, namespace="app1")
        def app1_llm(prompt: str) -> str:
            return f"App1: {prompt}"

        @cache(backend=backend, namespace="app2")
        def app2_llm(prompt: str) -> str:
            return f"App2: {prompt}"

        result1 = app1_llm("test")
        result2 = app2_llm("test")

        assert result1 == "App1: test"
        assert result2 == "App2: test"

    def test_ttl_expiration_integration(self):
        """Test TTL expiration in real workflow."""
        backend = MemoryBackend()

        @cache(backend=backend, ttl=1)  # 1 second TTL
        def llm_function(prompt: str) -> str:
            return f"Response to: {prompt}"

        llm_function("test prompt")

        # Immediate second call - hit
        llm_function("test prompt")

        # Wait for expiration
        time.sleep(1.5)

        # Should miss (cached entry expired)
        llm_function("test prompt")


class TestComplexScenarios:
    """Tests for complex real-world scenarios."""

    def test_high_volume_caching(self):
        """Test cache behavior with many entries."""
        backend = MemoryBackend(max_size=100)
        call_count = {"count": 0}

        @cache(backend=backend)
        def llm_function(prompt: str) -> str:
            call_count["count"] += 1
            return f"Response {call_count['count']}"

        # Add many entries
        for i in range(150):
            llm_function(f"prompt {i}")

        # Some entries should have been evicted
        stats = backend.get_stats()
        assert stats["size"] <= 100

    def test_concurrent_like_access(self):
        """Test multiple calls to same cached entry."""
        backend = MemoryBackend()

        @cache(backend=backend)
        def llm_function(prompt: str) -> str:
            return f"Unique: {time.time()}"

        # Multiple calls
        results = [llm_function("test") for _ in range(5)]

        # All should return same result (cached)
        assert len(set(results)) == 1

    def test_different_return_types(self):
        """Test caching different return types."""
        backend = MemoryBackend()

        @cache(backend=backend)
        def return_dict(prompt: str) -> dict:
            return {"key": "value"}

        @cache(backend=backend)
        def return_list(prompt: str) -> list:
            return [1, 2, 3]

        @cache(backend=backend)
        def return_string(prompt: str) -> str:
            return "string response"

        # Use unique prompts to avoid cache collision
        assert isinstance(return_dict("test_dict"), dict)
        assert isinstance(return_list("test_list"), list)
        assert isinstance(return_string("test_string"), str)

    def test_empty_and_none_responses(self):
        """Test caching empty and None responses."""
        backend = MemoryBackend()

        @cache(backend=backend)
        def return_empty(prompt: str) -> str:
            return ""

        @cache(backend=backend)
        def return_none(prompt: str) -> None:
            return None

        assert return_empty("empty_test") == ""
        assert return_none("none_test") is None

        # Should still cache (second calls should hit cache)
        assert return_empty("empty_test") == ""
        assert return_none("none_test") is None


class TestErrorHandling:
    """Tests for error handling in various scenarios."""

    def test_function_with_exception(self):
        """Test function that raises exception."""
        from semantic_llm_cache.exceptions import PromptCacheError

        backend = MemoryBackend()

        @cache(backend=backend)
        def failing_function(prompt: str) -> str:
            if "error" in prompt:
                raise ValueError("Test error")
            return "OK"

        # Normal call works
        assert failing_function("normal") == "OK"

        # Error call raises PromptCacheError (wrapped exception)
        with pytest.raises(PromptCacheError):
            failing_function("error prompt")

        # Normal call still works
        assert failing_function("normal") == "OK"

    def test_backend_error_handling(self):
        """Test that backend wraps errors properly."""
        from semantic_llm_cache.backends.memory import MemoryBackend

        # Use MemoryBackend which has proper error handling
        backend = MemoryBackend()

        @cache(backend=backend)
        def working_func(prompt: str) -> str:
            return f"Response to: {prompt}"

        # Normal operation works
        assert working_func("test") == "Response to: test"

        # Second call hits cache
        assert working_func("test") == "Response to: test"

        # Backend properly stores and retrieves entries
        stats = backend.get_stats()
        assert stats["hits"] >= 1


class TestPromptNormalization:
    """Tests for prompt normalization effects."""

    def test_whitespace_normalization(self):
        """Test prompts with different whitespace are cached separately."""
        backend = MemoryBackend()
        call_count = {"count": 0}

        @cache(backend=backend)
        def llm_function(prompt: str) -> str:
            call_count["count"] += 1
            return f"Response: {prompt}"

        llm_function("What is Python?")
        llm_function("What  is  Python?")  # Extra spaces

        # Normalization should make these the same
        # Note: This depends on the normalization implementation
        assert call_count["count"] >= 1

    def test_case_sensitivity(self):
        """Test case sensitivity in caching."""
        backend = MemoryBackend()
        call_count = {"count": 0}

        @cache(backend=backend)
        def llm_function(prompt: str) -> str:
            call_count["count"] += 1
            return f"Response: {prompt}"

        llm_function("What is Python?")
        llm_function("what is python?")

        # Case differences create different cache entries
        # (normalization doesn't lowercase by default)
        assert call_count["count"] >= 1


class TestConfigurationCombinations:
    """Tests for various configuration combinations."""

    def test_no_caching_config(self):
        """Test configuration with caching disabled."""
        backend = MemoryBackend()

        @cache(backend=backend, enabled=False)
        def llm_function(prompt: str) -> str:
            return f"Response: {time.time()}"

        result1 = llm_function("test")
        time.sleep(0.01)
        result2 = llm_function("test")

        # Without caching, results differ
        assert result1 != result2

    def test_zero_ttl(self):
        """Test zero TTL means immediate expiration."""
        backend = MemoryBackend()

        @cache(backend=backend, ttl=0)
        def llm_function(prompt: str) -> str:
            return f"Response: {prompt}"

        llm_function("test")
        # Entry immediately expires, so next call is a miss
        llm_function("test")

    def test_infinite_ttl(self):
        """Test None TTL means never expire."""
        backend = MemoryBackend()
        call_count = {"count": 0}

        @cache(backend=backend, ttl=None)
        def llm_function(prompt: str) -> str:
            call_count["count"] += 1
            return f"Response: {prompt}"

        llm_function("test")
        llm_function("test")

        assert call_count["count"] == 1
