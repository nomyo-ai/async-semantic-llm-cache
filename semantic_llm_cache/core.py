"""Core cache decorator and API for llm-semantic-cache."""

import functools
import inspect
import time
from typing import Any, Callable, Optional, ParamSpec, TypeVar

from semantic_llm_cache.backends import MemoryBackend
from semantic_llm_cache.backends.base import BaseBackend
from semantic_llm_cache.config import CacheConfig, CacheEntry
from semantic_llm_cache.exceptions import PromptCacheError
from semantic_llm_cache.similarity import EmbeddingCache
from semantic_llm_cache.stats import _stats_manager
from semantic_llm_cache.utils import hash_prompt, normalize_prompt

P = ParamSpec("P")
R = TypeVar("R")


def _extract_prompt(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Extract prompt string from function arguments."""
    if args and isinstance(args[0], str):
        return args[0]
    if "prompt" in kwargs:
        return str(kwargs["prompt"])
    return str(args) + str(sorted(kwargs.items()))


class CacheContext:
    """Context manager for cache configuration.

    Supports both sync (with) and async (async with) usage.

    Examples:
        >>> async with CacheContext(similarity=0.9) as ctx:
        ...     result = await llm_call("prompt")
        ...     print(ctx.stats)
    """

    def __init__(
        self,
        similarity: Optional[float] = None,
        ttl: Optional[int] = None,
        namespace: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        self._config = CacheConfig(
            similarity_threshold=similarity if similarity is not None else 1.0,
            ttl=ttl,
            namespace=namespace if namespace is not None else "default",
            enabled=enabled if enabled is not None else True,
        )
        self._stats: dict[str, Any] = {"hits": 0, "misses": 0}

    def __enter__(self) -> "CacheContext":
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    async def __aenter__(self) -> "CacheContext":
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    @property
    def stats(self) -> dict[str, Any]:
        return self._stats.copy()

    @property
    def config(self) -> CacheConfig:
        return self._config


class CachedLLM:
    """Wrapper class for LLM calls with automatic caching.

    Examples:
        >>> llm = CachedLLM(similarity=0.9)
        >>> response = await llm.achat("What is Python?", llm_func=my_async_llm)
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4",
        similarity: float = 1.0,
        ttl: Optional[int] = 3600,
        backend: Optional[BaseBackend] = None,
        namespace: str = "default",
        enabled: bool = True,
    ) -> None:
        self._provider = provider
        self._model = model
        self._backend = backend or MemoryBackend()
        self._embedding_cache = EmbeddingCache()
        self._config = CacheConfig(
            similarity_threshold=similarity,
            ttl=ttl,
            namespace=namespace,
            enabled=enabled,
        )

    async def achat(
        self,
        prompt: str,
        llm_func: Optional[Callable[[str], Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Get response with caching (async).

        Args:
            prompt: Input prompt
            llm_func: Async or sync LLM function to call on cache miss
            **kwargs: Additional arguments for llm_func

        Returns:
            LLM response (cached or fresh)
        """
        if llm_func is None:
            raise ValueError("llm_func is required for CachedLLM.achat()")

        @cache(
            similarity=self._config.similarity_threshold,
            ttl=self._config.ttl,
            backend=self._backend,
            namespace=self._config.namespace,
            enabled=self._config.enabled,
        )
        async def _cached_call(p: str) -> Any:
            result = llm_func(p, **kwargs)
            if inspect.isawaitable(result):
                return await result
            return result

        return await _cached_call(prompt)


def cache(
    similarity: float = 1.0,
    ttl: Optional[int] = 3600,
    backend: Optional[BaseBackend] = None,
    namespace: str = "default",
    enabled: bool = True,
    key_func: Optional[Callable[..., str]] = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator for caching LLM function responses.

    Auto-detects whether the decorated function is async or sync and returns
    the appropriate wrapper. Both variants share identical cache logic.

    Async functions get a true async wrapper (awaits all backend calls).
    Sync functions get a sync wrapper that drives the async backends via a
    temporary event loop — not suitable inside a running loop; prefer decorating
    async functions when integrating with async frameworks like FastAPI.

    Args:
        similarity: Cosine similarity threshold (1.0=exact, 0.9=semantic)
        ttl: Time-to-live in seconds (None=forever)
        backend: Async storage backend (None=in-memory)
        namespace: Cache namespace for isolation
        enabled: Whether caching is enabled
        key_func: Custom cache key function

    Returns:
        Decorated function with caching

    Examples:
        >>> @cache(similarity=0.9, ttl=3600)
        ... async def ask_llm(prompt: str) -> str:
        ...     return await call_ollama(prompt)

        >>> @cache()
        ... def ask_llm_sync(prompt: str) -> str:
        ...     return call_ollama_sync(prompt)
    """
    _backend = backend or MemoryBackend()
    embedding_cache = EmbeddingCache()

    def decorator(func: Callable[P, R]) -> Callable[P, R]:

        if inspect.iscoroutinefunction(func):
            # ── Async wrapper ────────────────────────────────────────────────
            @functools.wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                if not enabled:
                    return await func(*args, **kwargs)

                start_time = time.time()
                prompt = _extract_prompt(args, kwargs)  # type: ignore[arg-type]
                normalized = normalize_prompt(prompt)
                cache_key = (
                    key_func(*args, **kwargs)  # type: ignore[arg-type]
                    if key_func
                    else hash_prompt(normalized, namespace)
                )

                # 1. Exact match
                entry = await _backend.get(cache_key)
                if entry is not None:
                    latency_ms = (time.time() - start_time) * 1000
                    _stats_manager.record_hit(
                        namespace,
                        latency_saved_ms=latency_ms,
                        saved_cost=entry.estimate_cost(0.001, 0.002),
                    )
                    return entry.response  # type: ignore[return-value]

                # 2. Semantic match
                if similarity < 1.0:
                    query_embedding = await embedding_cache.aencode(normalized)
                    result = await _backend.find_similar(
                        query_embedding, threshold=similarity, namespace=namespace
                    )
                    if result is not None:
                        _, matched_entry, _ = result
                        latency_ms = (time.time() - start_time) * 1000
                        _stats_manager.record_hit(
                            namespace,
                            latency_saved_ms=latency_ms,
                            saved_cost=matched_entry.estimate_cost(0.001, 0.002),
                        )
                        return matched_entry.response  # type: ignore[return-value]

                # 3. Cache miss — call through
                _stats_manager.record_miss(namespace)

                try:
                    response = await func(*args, **kwargs)
                except Exception as e:
                    raise PromptCacheError(f"LLM function call failed: {e}") from e

                embedding = None
                if similarity < 1.0:
                    embedding = await embedding_cache.aencode(normalized)

                await _backend.set(
                    cache_key,
                    CacheEntry(
                        prompt=normalized,
                        response=response,
                        embedding=embedding,
                        created_at=time.time(),
                        ttl=ttl,
                        namespace=namespace,
                        hit_count=0,
                        input_tokens=len(normalized) // 4,
                        output_tokens=len(str(response)) // 4,
                    ),
                )
                return response  # type: ignore[return-value]

            return async_wrapper  # type: ignore[return-value]

        else:
            # ── Sync wrapper (backwards compatibility) ───────────────────────
            # Drives async backends via a dedicated event loop per call.
            # Do NOT use inside a running event loop (e.g. FastAPI handlers).
            import asyncio

            @functools.wraps(func)
            def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                if not enabled:
                    return func(*args, **kwargs)

                start_time = time.time()
                prompt = _extract_prompt(args, kwargs)  # type: ignore[arg-type]
                normalized = normalize_prompt(prompt)
                cache_key = (
                    key_func(*args, **kwargs)  # type: ignore[arg-type]
                    if key_func
                    else hash_prompt(normalized, namespace)
                )

                loop = asyncio.new_event_loop()
                try:
                    # 1. Exact match
                    entry = loop.run_until_complete(_backend.get(cache_key))
                    if entry is not None:
                        latency_ms = (time.time() - start_time) * 1000
                        _stats_manager.record_hit(
                            namespace,
                            latency_saved_ms=latency_ms,
                            saved_cost=entry.estimate_cost(0.001, 0.002),
                        )
                        return entry.response  # type: ignore[return-value]

                    # 2. Semantic match
                    if similarity < 1.0:
                        query_embedding = embedding_cache.encode(normalized)
                        result = loop.run_until_complete(
                            _backend.find_similar(
                                query_embedding, threshold=similarity, namespace=namespace
                            )
                        )
                        if result is not None:
                            _, matched_entry, _ = result
                            latency_ms = (time.time() - start_time) * 1000
                            _stats_manager.record_hit(
                                namespace,
                                latency_saved_ms=latency_ms,
                                saved_cost=matched_entry.estimate_cost(0.001, 0.002),
                            )
                            return matched_entry.response  # type: ignore[return-value]

                    # 3. Cache miss
                    _stats_manager.record_miss(namespace)

                    try:
                        response = func(*args, **kwargs)
                    except Exception as e:
                        raise PromptCacheError(f"LLM function call failed: {e}") from e

                    embedding = None
                    if similarity < 1.0:
                        embedding = embedding_cache.encode(normalized)

                    loop.run_until_complete(
                        _backend.set(
                            cache_key,
                            CacheEntry(
                                prompt=normalized,
                                response=response,
                                embedding=embedding,
                                created_at=time.time(),
                                ttl=ttl,
                                namespace=namespace,
                                hit_count=0,
                                input_tokens=len(normalized) // 4,
                                output_tokens=len(str(response)) // 4,
                            ),
                        )
                    )
                    return response  # type: ignore[return-value]
                finally:
                    loop.close()

            return sync_wrapper  # type: ignore[return-value]

    return decorator


# Global default backend for utility functions
_default_backend: Optional[BaseBackend] = None


def get_default_backend() -> BaseBackend:
    """Get default storage backend."""
    global _default_backend
    if _default_backend is None:
        _default_backend = MemoryBackend()
    return _default_backend


def set_default_backend(backend: BaseBackend) -> None:
    """Set default storage backend."""
    global _default_backend
    _default_backend = backend
    _stats_manager.set_backend(backend)


__all__ = [
    "cache",
    "CacheContext",
    "CachedLLM",
    "get_default_backend",
    "set_default_backend",
]
