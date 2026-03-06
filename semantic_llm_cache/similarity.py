"""Embedding generation and similarity matching for llm-semantic-cache."""

import asyncio
import hashlib
from functools import lru_cache
from typing import Optional

import numpy as np

from semantic_llm_cache.exceptions import PromptCacheError


class EmbeddingProvider:
    """Base class for embedding providers."""

    def encode(self, text: str) -> list[float]:
        """Generate embedding for text.

        Args:
            text: Input text to encode

        Returns:
            Embedding vector as list of floats
        """
        raise NotImplementedError


class DummyEmbeddingProvider(EmbeddingProvider):
    """Fallback embedding provider using hash-based vectors.

    Provides consistent embeddings without external dependencies.
    Not semantically meaningful but provides consistent cache keys.
    """

    def __init__(self, dim: int = 384) -> None:
        """Initialize dummy provider.

        Args:
            dim: Embedding dimension (matches MiniLM default)
        """
        self._dim = dim

    def encode(self, text: str) -> list[float]:
        """Generate hash-based embedding for text.

        Args:
            text: Input text to encode

        Returns:
            Deterministic embedding vector based on text hash
        """
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()

        values = np.frombuffer(hash_bytes, dtype=np.uint8)[: self._dim].astype(
            np.float32
        )

        if len(values) < self._dim:
            values = np.pad(values, (0, self._dim - len(values)))

        norm = np.linalg.norm(values)
        if norm > 0:
            values = values / norm

        return values.tolist()


class SentenceTransformerProvider(EmbeddingProvider):
    """Sentence-transformers based embedding provider.

    Uses local models like MiniLM for semantic embeddings.
    Inference is CPU/GPU-bound; use aencode() from async contexts.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize sentence-transformer provider.

        Args:
            model_name: Name of sentence-transformer model
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise PromptCacheError(
                "sentence-transformers package required for semantic matching. "
                "Install with: pip install semantic-llm-cache[semantic]"
            ) from e

        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    def encode(self, text: str) -> list[float]:
        """Generate embedding for text (blocking — use aencode from async code).

        Args:
            text: Input text to encode

        Returns:
            Normalized embedding vector
        """
        embedding = self._model.encode(text, convert_to_numpy=True)
        embedding = np.asarray(embedding, dtype=np.float32)

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.tolist()


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI API-based embedding provider.

    Uses OpenAI's embedding API for high-quality semantic embeddings.
    Network I/O — always use aencode() from async contexts.
    """

    def __init__(
        self, api_key: Optional[str] = None, model: str = "text-embedding-3-small"
    ) -> None:
        """Initialize OpenAI embedding provider.

        Args:
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if None)
            model: OpenAI embedding model to use
        """
        try:
            import openai
        except ImportError as e:
            raise PromptCacheError(
                "openai package required for OpenAI embeddings. "
                "Install with: pip install semantic-llm-cache[openai]"
            ) from e

        self._client = openai.OpenAI(api_key=api_key)
        self._model = model

    def encode(self, text: str) -> list[float]:
        """Generate embedding for text (blocking — use aencode from async code).

        Args:
            text: Input text to encode

        Returns:
            OpenAI embedding vector (already normalized)
        """
        response = self._client.embeddings.create(input=text, model=self._model)
        embedding = response.data[0].embedding

        embedding_arr = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding_arr)
        if norm > 0:
            embedding_arr = embedding_arr / norm

        return embedding_arr.tolist()


def cosine_similarity(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    """Calculate cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Similarity score between 0 and 1

    Raises:
        ValueError: If vectors have different dimensions
    """
    a_arr = np.asarray(a, dtype=np.float32)
    b_arr = np.asarray(b, dtype=np.float32)

    if a_arr.shape != b_arr.shape:
        raise ValueError(
            f"Vector dimension mismatch: {a_arr.shape} != {b_arr.shape}"
        )

    dot_product = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot_product / (norm_a * norm_b))


def _encode_with_provider(text: str, provider: EmbeddingProvider) -> tuple[float, ...]:
    """Helper function for LRU cache encoding.

    Args:
        text: Input text
        provider: Embedding provider

    Returns:
        Embedding as tuple for hashability
    """
    return tuple(provider.encode(text))


class EmbeddingCache:
    """Cache for embedding generation with LRU eviction.

    Use encode() from sync contexts, aencode() from async contexts.
    aencode() offloads blocking inference to a thread pool via asyncio.to_thread.
    """

    def __init__(
        self,
        provider: Optional[EmbeddingProvider] = None,
        cache_size: int = 1024,
    ) -> None:
        """Initialize embedding cache.

        Args:
            provider: Embedding provider (uses DummyEmbeddingProvider if None)
            cache_size: Maximum number of embeddings to cache
        """
        self._provider = provider or DummyEmbeddingProvider()
        self._cache_size = cache_size
        self._get_cached = lru_cache(maxsize=cache_size)(_encode_with_provider)

    def encode(self, text: str) -> list[float]:
        """Generate embedding with LRU caching (sync, blocking).

        Args:
            text: Input text to encode

        Returns:
            Embedding vector
        """
        return list(self._get_cached(text, self._provider))

    async def aencode(self, text: str) -> list[float]:
        """Generate embedding with LRU caching (async, non-blocking).

        CPU/network-bound work is offloaded to the default thread pool via
        asyncio.to_thread, keeping the event loop free.

        Args:
            text: Input text to encode

        Returns:
            Embedding vector
        """
        return await asyncio.to_thread(self.encode, text)

    def clear_cache(self) -> None:
        """Clear the embedding LRU cache."""
        self._get_cached.cache_clear()


def create_embedding_provider(
    provider_type: str = "auto",
    model_name: Optional[str] = None,
) -> EmbeddingProvider:
    """Create embedding provider based on type.

    Args:
        provider_type: Type of provider ("auto", "sentence-transformer", "openai", "dummy")
        model_name: Optional model name to use

    Returns:
        EmbeddingProvider instance
    """
    if provider_type == "auto":
        try:
            return SentenceTransformerProvider(model_name or "all-MiniLM-L6-v2")
        except PromptCacheError:
            return DummyEmbeddingProvider()

    if provider_type == "sentence-transformer":
        return SentenceTransformerProvider(model_name or "all-MiniLM-L6-v2")

    if provider_type == "openai":
        return OpenAIEmbeddingProvider(model=model_name)

    if provider_type == "dummy":
        return DummyEmbeddingProvider()

    raise ValueError(f"Unknown provider type: {provider_type}")
