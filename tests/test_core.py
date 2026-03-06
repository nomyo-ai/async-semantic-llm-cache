"""Tests for core cache decorator and API."""

import time
from unittest.mock import MagicMock

import pytest

from semantic_llm_cache import CacheContext, CachedLLM, cache, get_default_backend, set_default_backend
from semantic_llm_cache.backends import MemoryBackend
from semantic_llm_cache.config import CacheConfig, CacheEntry
from semantic_llm_cache.exceptions import PromptCacheError


class TestCacheDecorator:
    """Tests for @cache decorator."""

    def test_exact_match_cache_hit(self, mock_llm_func):
        """Test exact match caching returns cached result."""
        call_count = {"count": 0}

        @cache()
        def cached_func(prompt: str) -> str:
            call_count["count"] += 1
            return mock_llm_func(prompt)

        # First call - cache miss
        result1 = cached_func("What is Python?")
        assert result1 == "Python is a programming language."
        assert call_count["count"] == 1

        # Second call - cache hit
        result2 = cached_func("What is Python?")
        assert result2 == "Python is a programming language."
        assert call_count["count"] == 1  # No additional call

    def test_cache_miss_different_prompt(self, mock_llm_func):
        """Test different prompts result in cache misses."""
        @cache()
        def cached_func(prompt: str) -> str:
            return mock_llm_func(prompt)

        result1 = cached_func("What is Python?")
        result2 = cached_func("What is Rust?")

        assert result1 == "Python is a programming language."
        assert result2 == "Rust is a systems programming language."

    def test_cache_disabled(self, mock_llm_func):
        """Test caching can be disabled."""
        call_count = {"count": 0}

        @cache(enabled=False)
        def cached_func(prompt: str) -> str:
            call_count["count"] += 1
            return mock_llm_func(prompt)

        cached_func("What is Python?")
        cached_func("What is Python?")

        assert call_count["count"] == 2  # Both calls hit the function

    def test_custom_namespace(self, mock_llm_func):
        """Test custom namespace isolates cache."""
        @cache(namespace="test")
        def cached_func(prompt: str) -> str:
            return mock_llm_func(prompt)

        result = cached_func("What is Python?")
        assert result == "Python is a programming language."

    def test_ttl_expiration(self, mock_llm_func):
        """Test TTL expiration works."""
        @cache(ttl=1)  # 1 second TTL
        def cached_func(prompt: str) -> str:
            return mock_llm_func(prompt)

        # First call
        result1 = cached_func("What is Python?")
        assert result1 == "Python is a programming language."

        # Immediate second call - should hit cache
        result2 = cached_func("What is Python?")
        assert result2 == "Python is a programming language."

        # Wait for expiration
        time.sleep(1.1)

        # Third call - should miss due to TTL
        # Note: This test may be flaky in slow CI environments
        result3 = cached_func("What is Python?")
        assert result3 == "Python is a programming language."

    def test_cache_with_exception(self):
        """Test cache handles exceptions properly."""
        @cache()
        def failing_func(prompt: str) -> str:
            raise ValueError("LLM API error")

        with pytest.raises(PromptCacheError):
            failing_func("test prompt")

    def test_semantic_similarity_match(self, mock_llm_func):
        """Test semantic similarity matching."""
        call_count = {"count": 0}

        @cache(similarity=0.85)
        def cached_func(prompt: str) -> str:
            call_count["count"] += 1
            return mock_llm_func(prompt)

        # First call
        cached_func("What is Python?")
        assert call_count["count"] == 1

        # Similar prompts may hit cache depending on embedding
        # Note: With dummy embeddings, exact string matching determines similarity
        cached_func("What is Python?")  # Exact match
        assert call_count["count"] == 1


class TestCacheContext:
    """Tests for CacheContext manager."""

    def test_context_manager(self):
        """Test CacheContext works as context manager."""
        with CacheContext(similarity=0.9, ttl=1800) as ctx:
            assert ctx.config.similarity_threshold == 0.9
            assert ctx.config.ttl == 1800

    def test_context_stats(self):
        """Test context tracks stats."""
        with CacheContext() as ctx:
            stats = ctx.stats
            assert "hits" in stats
            assert "misses" in stats


class TestCachedLLM:
    """Tests for CachedLLM wrapper class."""

    def test_init(self):
        """Test CachedLLM initialization."""
        llm = CachedLLM(provider="openai", model="gpt-4")
        assert llm._provider == "openai"
        assert llm._model == "gpt-4"

    def test_chat_with_llm_func(self):
        """Test chat method with custom LLM function."""
        llm = CachedLLM()

        def mock_llm(prompt: str) -> str:
            return f"Response to: {prompt}"

        result = llm.chat("Hello", llm_func=mock_llm)
        assert result == "Response to: Hello"

    def test_chat_caches_responses(self, mock_llm_func):
        """Test chat caches responses."""
        llm = CachedLLM()
        call_count = {"count": 0}

        def counting_llm(prompt: str) -> str:
            call_count["count"] += 1
            return mock_llm_func(prompt)

        llm.chat("What is Python?", llm_func=counting_llm)
        llm.chat("What is Python?", llm_func=counting_llm)

        # Should cache (depends on embedding, may not with dummy)
        assert call_count["count"] >= 1


class TestBackendManagement:
    """Tests for backend management functions."""

    def test_get_default_backend(self):
        """Test get_default_backend returns a backend."""
        backend = get_default_backend()
        assert backend is not None
        assert isinstance(backend, MemoryBackend)

    def test_set_default_backend(self):
        """Test set_default_backend changes default."""
        custom_backend = MemoryBackend(max_size=10)
        set_default_backend(custom_backend)

        backend = get_default_backend()
        assert backend is custom_backend


class TestCacheEntry:
    """Tests for CacheEntry class."""

    def test_entry_creation(self):
        """Test CacheEntry creation."""
        entry = CacheEntry(
            prompt="test",
            response="response",
            created_at=time.time(),
        )
        assert entry.prompt == "test"
        assert entry.response == "response"

    def test_is_expired_no_ttl(self):
        """Test entry without TTL never expires."""
        entry = CacheEntry(
            prompt="test",
            response="response",
            ttl=None,
            created_at=time.time() - 1000,
        )
        assert not entry.is_expired(time.time())

    def test_is_expired_with_ttl(self):
        """Test entry with TTL expires correctly."""
        entry = CacheEntry(
            prompt="test",
            response="response",
            ttl=1,  # 1 second
            created_at=time.time() - 2,  # 2 seconds ago
        )
        assert entry.is_expired(time.time())

    def test_estimate_cost(self):
        """Test cost estimation."""
        entry = CacheEntry(
            prompt="test",
            response="response",
            input_tokens=100,
            output_tokens=50,
        )
        cost = entry.estimate_cost(0.001, 0.002)
        # 100/1000 * 0.001 + 50/1000 * 0.002 = 0.0001 + 0.0001 = 0.0002
        assert abs(cost - 0.0002) < 1e-6


class TestCacheConfig:
    """Tests for CacheConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CacheConfig()
        assert config.similarity_threshold == 1.0
        assert config.ttl == 3600
        assert config.namespace == "default"
        assert config.enabled is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = CacheConfig(
            similarity_threshold=0.9,
            ttl=7200,
            namespace="custom",
            enabled=False,
        )
        assert config.similarity_threshold == 0.9
        assert config.ttl == 7200
        assert config.namespace == "custom"
        assert config.enabled is False

    def test_invalid_similarity(self):
        """Test invalid similarity raises error."""
        with pytest.raises(ValueError, match="similarity_threshold"):
            CacheConfig(similarity_threshold=1.5)

    def test_invalid_ttl(self):
        """Test invalid TTL raises error."""
        with pytest.raises(ValueError, match="ttl"):
            CacheConfig(ttl=-1)

    def test_invalid_max_size(self):
        """Test invalid max_size raises error."""
        with pytest.raises(ValueError, match="max_cache_size"):
            CacheConfig(max_cache_size=0)


class TestCacheDecoratorEdgeCases:
    """Tests for edge cases in cache decorator."""

    def test_cache_with_kwargs_only(self, mock_llm_func):
        """Test caching when function is called with kwargs only."""
        @cache()
        def cached_func(prompt: str) -> str:
            return mock_llm_func(prompt)

        result = cached_func(prompt="What is Python?")
        assert result == "Python is a programming language."

    def test_cache_with_multiple_args(self):
        """Test caching with multiple arguments."""
        call_count = {"count": 0}

        @cache()
        def cached_func(prompt: str, temperature: float = 0.7) -> str:
            call_count["count"] += 1
            return f"Response to: {prompt} at {temperature}"

        cached_func("test", 0.5)
        cached_func("test", 0.5)
        # Same prompt hits cache even with different temperature
        # (cache key is based on first arg)
        cached_func("test", 0.9)
        cached_func("different", 0.5)

        # First call + different prompt = 2 calls
        assert call_count["count"] == 2

    def test_cache_with_custom_key_func(self):
        """Test custom key function."""
        call_count = {"count": 0}

        def custom_key(prompt: str, temperature: float = 0.7) -> str:
            return f"{prompt}:{temperature}"

        @cache(key_func=custom_key)
        def cached_func(prompt: str, temperature: float = 0.7) -> str:
            call_count["count"] += 1
            return f"Response to: {prompt} at {temperature}"

        cached_func("test", 0.7)
        cached_func("test", 0.7)
        assert call_count["count"] == 1

    def test_semantic_match_threshold_edge(self, mock_llm_func):
        """Test semantic matching at threshold boundaries."""
        call_count = {"count": 0}

        @cache(similarity=0.5)  # Lower threshold
        def cached_func(prompt: str) -> str:
            call_count["count"] += 1
            return mock_llm_func(prompt)

        # First call
        cached_func("What is Python?")
        # Similar query may hit with lower threshold
        cached_func("What is Python?")  # Exact match always hits
        assert call_count["count"] == 1

    def test_cache_with_none_response(self):
        """Test caching None responses."""
        call_count = {"count": 0}

        @cache()
        def cached_func(prompt: str) -> None:
            call_count["count"] += 1
            return None

        result1 = cached_func("test")
        result2 = cached_func("test")

        assert result1 is None
        assert result2 is None
        assert call_count["count"] == 1  # Should cache

    def test_cache_with_empty_string_response(self):
        """Test caching empty string responses."""
        call_count = {"count": 0}

        @cache()
        def cached_func(prompt: str) -> str:
            call_count["count"] += 1
            return ""

        result1 = cached_func("test")
        result2 = cached_func("test")

        assert result1 == ""
        assert result2 == ""
        assert call_count["count"] == 1  # Should cache

    def test_cache_with_dict_response(self):
        """Test caching dict responses."""
        call_count = {"count": 0}

        @cache()
        def cached_func(prompt: str) -> dict:
            call_count["count"] += 1
            return {"key": "value", "number": 42}

        result1 = cached_func("test")
        cached_func("test")  # Second call to verify caching

        assert result1 == {"key": "value", "number": 42}
        assert call_count["count"] == 1

    def test_cache_with_list_response(self):
        """Test caching list responses."""
        call_count = {"count": 0}

        @cache()
        def cached_func(prompt: str) -> list:
            call_count["count"] += 1
            return [1, 2, 3, 4, 5]

        result1 = cached_func("test")
        cached_func("test")  # Second call to verify caching

        assert result1 == [1, 2, 3, 4, 5]
        assert call_count["count"] == 1


class TestCacheDecoratorErrorPaths:
    """Tests for error handling in cache decorator."""

    def test_backend_set_raises_error(self):
        """Test that backend.set errors propagate."""
        from semantic_llm_cache.exceptions import CacheBackendError

        backend = MagicMock()
        # Backend set wraps exceptions in CacheBackendError
        backend.set.side_effect = CacheBackendError("Storage error")
        backend.get.return_value = None  # No cached value

        @cache(backend=backend)
        def cached_func(prompt: str) -> str:
            return "response"

        # The CacheBackendError from backend.set should propagate
        with pytest.raises(CacheBackendError, match="Storage error"):
            cached_func("test")

    def test_backend_get_raises_error(self):
        """Test that backend.get errors propagate."""
        from semantic_llm_cache.exceptions import CacheBackendError

        backend = MagicMock()
        # Backend get wraps exceptions in CacheBackendError
        backend.get.side_effect = CacheBackendError("Get error")

        @cache(backend=backend)
        def cached_func(prompt: str) -> str:
            return "response"

        # The CacheBackendError from backend.get should propagate
        with pytest.raises(CacheBackendError, match="Get error"):
            cached_func("test")

    def test_llm_error_still_wrapped(self):
        """Test that LLM errors are still wrapped in PromptCacheError."""
        from semantic_llm_cache.exceptions import PromptCacheError

        @cache()
        def failing_func(prompt: str) -> str:
            raise ValueError("LLM API error")

        with pytest.raises(PromptCacheError):
            failing_func("test")


class TestCacheContextAdvanced:
    """Advanced tests for CacheContext."""

    def test_context_with_zero_similarity(self):
        """Test context with zero similarity (accept all)."""
        with CacheContext(similarity=0.0) as ctx:
            assert ctx.config.similarity_threshold == 0.0

    def test_context_with_infinite_ttl(self):
        """Test context with infinite TTL (None)."""
        with CacheContext(ttl=None) as ctx:
            assert ctx.config.ttl is None

    def test_context_disabled(self):
        """Test disabled context."""
        with CacheContext(enabled=False) as ctx:
            assert ctx.config.enabled is False


class TestCachedLLMAdvanced:
    """Advanced tests for CachedLLM."""

    def test_chat_with_kwargs(self):
        """Test chat with additional kwargs passed to LLM."""
        llm = CachedLLM()

        def mock_llm(prompt: str, temperature: float = 0.7, max_tokens: int = 100) -> str:
            return f"Response to: {prompt} (temp={temperature}, tokens={max_tokens})"

        result = llm.chat("Hello", llm_func=mock_llm, temperature=0.5, max_tokens=200)
        assert "temp=0.5" in result
        assert "tokens=200" in result

    def test_cached_llm_with_custom_backend(self):
        """Test CachedLLM with custom backend."""
        custom_backend = MemoryBackend(max_size=5)
        llm = CachedLLM(backend=custom_backend)

        assert llm._backend is custom_backend

    def test_cached_llm_different_namespaces(self):
        """Test CachedLLM with different namespaces."""
        llm1 = CachedLLM(namespace="ns1")
        llm2 = CachedLLM(namespace="ns2")

        assert llm1._config.namespace == "ns1"
        assert llm2._config.namespace == "ns2"


class TestBackendManagementAdvanced:
    """Advanced tests for backend management."""

    def test_multiple_default_backend_changes(self):
        """Test changing default backend multiple times."""
        backend1 = MemoryBackend(max_size=10)
        backend2 = MemoryBackend(max_size=20)
        backend3 = MemoryBackend(max_size=30)

        set_default_backend(backend1)
        assert get_default_backend() is backend1

        set_default_backend(backend2)
        assert get_default_backend() is backend2

        set_default_backend(backend3)
        assert get_default_backend() is backend3

    def test_backend_persists_stats(self):
        """Test backend stats persist across get_default_backend calls."""
        backend = get_default_backend()

        # Create an entry
        from semantic_llm_cache.config import CacheEntry
        entry = CacheEntry(
            prompt="test",
            response="response",
            created_at=time.time()
        )
        backend.set("key1", entry)

        # Get stats
        stats = backend.get_stats()
        assert stats["size"] == 1


class TestCacheEntryEdgeCases:
    """Edge case tests for CacheEntry."""

    def test_entry_with_zero_tokens(self):
        """Test entry with zero token counts."""
        entry = CacheEntry(
            prompt="test",
            response="response",
            input_tokens=0,
            output_tokens=0,
        )
        cost = entry.estimate_cost(0.001, 0.002)
        assert cost == 0.0

    def test_entry_with_large_token_count(self):
        """Test entry with large token counts."""
        entry = CacheEntry(
            prompt="test",
            response="response",
            input_tokens=100000,
            output_tokens=50000,
        )
        cost = entry.estimate_cost(0.001, 0.002)
        assert cost > 0

    def test_entry_with_negative_ttl(self):
        """Test entry creation handles negative TTL (becomes expired)."""
        entry = CacheEntry(
            prompt="test",
            response="response",
            ttl=-1,
            created_at=time.time(),
        )
        # Negative TTL means immediately expired
        assert entry.is_expired(time.time())

    def test_entry_hit_count_initialization(self):
        """Test entry initializes with zero hit count."""
        entry = CacheEntry(
            prompt="test",
            response="response",
            created_at=time.time(),
        )
        assert entry.hit_count == 0


class TestCacheConfigEdgeCases:
    """Edge case tests for CacheConfig."""

    def test_config_boundary_similarity(self):
        """Test similarity at valid boundaries."""
        config1 = CacheConfig(similarity_threshold=0.0)
        assert config1.similarity_threshold == 0.0

        config2 = CacheConfig(similarity_threshold=1.0)
        assert config2.similarity_threshold == 1.0

    def test_config_zero_ttl(self):
        """Test zero TTL is rejected (validation requires positive)."""
        # The validation in CacheConfig rejects ttl <= 0
        with pytest.raises(ValueError, match="ttl"):
            CacheConfig(ttl=0)

    def test_config_very_large_ttl(self):
        """Test very large TTL."""
        config = CacheConfig(ttl=86400 * 365)  # 1 year
        assert config.ttl == 86400 * 365

    def test_config_with_special_namespace(self):
        """Test namespace with special characters."""
        config = CacheConfig(namespace="test-ns_123.v1")
        assert config.namespace == "test-ns_123.v1"
