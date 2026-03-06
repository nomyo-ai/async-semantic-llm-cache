"""Pytest configuration and fixtures for prompt-cache tests."""

import time

import pytest

from semantic_llm_cache.backends import MemoryBackend
from semantic_llm_cache.config import CacheConfig, CacheEntry


@pytest.fixture
def backend():
    """Provide a fresh memory backend for each test."""
    return MemoryBackend()


@pytest.fixture
def cache_config():
    """Provide default cache configuration."""
    return CacheConfig()


@pytest.fixture
def sample_entry():
    """Provide a sample cache entry."""
    return CacheEntry(
        prompt="What is Python?",
        response="Python is a programming language.",
        embedding=[0.1, 0.2, 0.3],
        created_at=time.time(),  # Use current time
        ttl=3600,
        namespace="default",
        hit_count=0,
    )


@pytest.fixture
def mock_llm_func():
    """Provide a mock LLM function."""
    responses = {
        "What is Python?": "Python is a programming language.",
        "What's Python?": "Python is a programming language.",
        "Explain Python": "Python is a high-level programming language.",
        "What is Rust?": "Rust is a systems programming language.",
    }

    def _func(prompt: str) -> str:
        return responses.get(prompt, f"Response to: {prompt}")

    return _func
