"""Custom exceptions for prompt-cache."""


class PromptCacheError(Exception):
    """Base exception for prompt-cache errors."""

    pass


class CacheBackendError(PromptCacheError):
    """Exception raised when backend operations fail."""

    pass


class CacheSerializationError(PromptCacheError):
    """Exception raised when serialization/deserialization fails."""

    pass


class CacheNotFoundError(PromptCacheError):
    """Exception raised when cache entry is not found."""

    pass
