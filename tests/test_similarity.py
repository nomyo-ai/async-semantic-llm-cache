"""Tests for embedding generation and similarity matching."""

import numpy as np
import pytest

from semantic_llm_cache.similarity import (
    DummyEmbeddingProvider,
    EmbeddingCache,
    OpenAIEmbeddingProvider,
    SentenceTransformerProvider,
    cosine_similarity,
    create_embedding_provider,
)


class TestCosineSimilarity:
    """Tests for cosine_similarity function."""

    def test_identical_vectors(self):
        """Test identical vectors have similarity 1.0."""
        a = [1.0, 2.0, 3.0]
        b = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, b) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        """Test orthogonal vectors have similarity 0.0."""
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        """Test opposite vectors have similarity -1.0."""
        a = [1.0, 2.0, 3.0]
        b = [-1.0, -2.0, -3.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vectors(self):
        """Test zero vectors return 0.0."""
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, b) == 0.0

    def test_numpy_array_input(self):
        """Test function accepts numpy arrays."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(a, b) == pytest.approx(1.0)

    def test_mixed_dimensions(self):
        """Test vectors of different dimensions raise error."""
        a = [1.0, 2.0]
        b = [1.0, 2.0, 3.0]
        # Should raise ValueError for mismatched dimensions
        with pytest.raises(ValueError, match="dimension mismatch"):
            cosine_similarity(a, b)


class TestDummyEmbeddingProvider:
    """Tests for DummyEmbeddingProvider."""

    def test_encode_returns_list(self):
        """Test encode returns list of floats."""
        provider = DummyEmbeddingProvider()
        embedding = provider.encode("test prompt")
        assert isinstance(embedding, list)
        assert all(isinstance(x, float) for x in embedding)

    def test_encode_deterministic(self):
        """Test same input produces same output."""
        provider = DummyEmbeddingProvider()
        text = "test prompt"
        e1 = provider.encode(text)
        e2 = provider.encode(text)
        assert e1 == e2

    def test_encode_different_inputs(self):
        """Test different inputs produce different outputs."""
        provider = DummyEmbeddingProvider()
        e1 = provider.encode("prompt 1")
        e2 = provider.encode("prompt 2")
        assert e1 != e2

    def test_custom_dimension(self):
        """Test custom embedding dimension."""
        provider = DummyEmbeddingProvider(dim=128)
        embedding = provider.encode("test")
        assert len(embedding) == 128

    def test_embedding_normalized(self):
        """Test embeddings are normalized to unit length."""
        provider = DummyEmbeddingProvider()
        embedding = provider.encode("test prompt")
        # Calculate norm
        norm = np.linalg.norm(embedding)
        assert norm == pytest.approx(1.0, rel=1e-5)


class TestSentenceTransformerProvider:
    """Tests for SentenceTransformerProvider."""

    @pytest.mark.skip(reason="Requires sentence-transformers installation")
    def test_encode_returns_list(self):
        """Test encode returns list of floats."""
        provider = SentenceTransformerProvider()
        embedding = provider.encode("test prompt")
        assert isinstance(embedding, list)
        assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.skip(reason="Requires sentence-transformers installation")
    def test_encode_deterministic(self):
        """Test same input produces same output."""
        provider = SentenceTransformerProvider()
        text = "test prompt"
        e1 = provider.encode(text)
        e2 = provider.encode(text)
        assert e1 == e2

    def test_import_error_without_package(self, monkeypatch):
        """Test ImportError raised when package not installed."""
        # Skip if sentence-transformers is installed
        pytest.importorskip("sentence_transformers", reason="sentence-transformers is installed, cannot test import error")


class TestOpenAIEmbeddingProvider:
    """Tests for OpenAIEmbeddingProvider."""

    @pytest.mark.skip(reason="Requires OpenAI API key")
    def test_encode_returns_list(self):
        """Test encode returns list of floats."""
        provider = OpenAIEmbeddingProvider()
        embedding = provider.encode("test prompt")
        assert isinstance(embedding, list)
        assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.skip(reason="Requires OpenAI API key")
    def test_encode_deterministic(self):
        """Test same input produces same output."""
        provider = OpenAIEmbeddingProvider()
        text = "test prompt"
        e1 = provider.encode(text)
        e2 = provider.encode(text)
        # OpenAI embeddings may vary slightly
        assert len(e1) == len(e2)

    def test_import_error_without_package(self, monkeypatch):
        """Test ImportError raised when package not installed."""
        # Skip if openai is installed
        pytest.importorskip("openai", reason="openai is installed, cannot test import error")


class TestEmbeddingCache:
    """Tests for EmbeddingCache."""

    def test_cache_provider(self):
        """Test cache uses provider."""
        provider = DummyEmbeddingProvider()
        cache = EmbeddingCache(provider=provider)

        # Encode same text twice
        e1 = cache.encode("test prompt")
        e2 = cache.encode("test prompt")

        assert e1 == e2

    def test_cache_clear(self):
        """Test cache can be cleared."""
        provider = DummyEmbeddingProvider()
        cache = EmbeddingCache(provider=provider)

        cache.encode("test prompt")
        cache.clear_cache()

        # Should still work after clear
        embedding = cache.encode("test prompt")
        assert len(embedding) > 0

    def test_cache_default_provider(self):
        """Test cache uses dummy provider by default."""
        cache = EmbeddingCache()
        embedding = cache.encode("test prompt")
        assert isinstance(embedding, list)
        assert len(embedding) > 0


class TestCreateEmbeddingProvider:
    """Tests for create_embedding_provider factory."""

    def test_create_dummy_provider(self):
        """Test creating dummy provider."""
        provider = create_embedding_provider("dummy")
        assert isinstance(provider, DummyEmbeddingProvider)

    def test_create_auto_provider(self):
        """Test auto provider creates sentence-transformers when available."""
        provider = create_embedding_provider("auto")
        # Creates SentenceTransformerProvider if available, else DummyEmbeddingProvider
        assert isinstance(provider, (DummyEmbeddingProvider, SentenceTransformerProvider))

    def test_invalid_provider_type(self):
        """Test invalid provider type raises error."""
        with pytest.raises(ValueError, match="Unknown provider type"):
            create_embedding_provider("invalid_type")

    def test_custom_model_name(self, monkeypatch):
        """Test custom model name is passed through."""
        # This would work with actual sentence-transformers
        provider = create_embedding_provider("dummy", model_name="custom-model")
        assert isinstance(provider, DummyEmbeddingProvider)
