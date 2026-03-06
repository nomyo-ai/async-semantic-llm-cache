"""Configuration management for prompt-cache."""

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class CacheConfig:
    """Configuration for cache behavior."""

    similarity_threshold: float = 1.0  # 1.0 = exact match, lower = semantic
    ttl: Optional[int] = 3600  # Time to live in seconds, None = forever
    namespace: str = "default"  # Isolate different use cases
    enabled: bool = True  # Enable/disable caching
    key_func: Optional[Callable[[Any], str]] = None  # Custom cache key function

    # Cost estimation for statistics (USD per 1K tokens)
    input_cost_per_1k: float = 0.001  # Default ~$1/1M for cheaper models
    output_cost_per_1k: float = 0.002  # Default ~$2/1M for cheaper models

    # Performance settings
    max_cache_size: Optional[int] = None  # LRU eviction when set
    embedding_model: str = "all-MiniLM-L6-v2"  # Default sentence-transformer model

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        if self.ttl is not None and self.ttl <= 0:
            raise ValueError("ttl must be positive or None")
        if self.max_cache_size is not None and self.max_cache_size <= 0:
            raise ValueError("max_cache_size must be positive or None")


@dataclass
class CacheEntry:
    """A cached response with metadata."""

    prompt: str
    response: Any
    embedding: Optional[list[float]] = None  # Normalized embedding vector
    created_at: float = 0.0  # Unix timestamp
    ttl: Optional[int] = None  # Time to live in seconds
    namespace: str = "default"
    hit_count: int = 0

    # Approximate token counts for cost estimation
    input_tokens: int = 0
    output_tokens: int = 0

    def is_expired(self, current_time: float) -> bool:
        """Check if entry has expired based on TTL."""
        if self.ttl is None:
            return False
        return (current_time - self.created_at) > self.ttl

    def estimate_cost(self, input_cost: float, output_cost: float) -> float:
        """Estimate cost savings in USD."""
        input_savings = (self.input_tokens / 1000) * input_cost
        output_savings = (self.output_tokens / 1000) * output_cost
        return input_savings + output_savings
