"""Utility functions for prompt-cache."""

import hashlib
import re
from typing import Any


def normalize_prompt(prompt: str) -> str:
    """Normalize prompt text for consistent caching.

    Args:
        prompt: Raw prompt text

    Returns:
        Normalized prompt text
    """
    # Remove extra whitespace
    prompt = " ".join(prompt.split())

    # Lowercase for better matching (optional - can affect semantics)
    # prompt = prompt.lower()

    # Remove common filler words at start
    filler_pattern = r"^(please|can you|could you|i need|i want)\s+"
    prompt = re.sub(filler_pattern, "", prompt, flags=re.IGNORECASE)

    # Normalize quotes
    prompt = prompt.replace('"', "'").replace("`", "'")

    # Remove trailing punctuation
    prompt = prompt.rstrip("?!.")

    return prompt.strip()


def hash_prompt(prompt: str, namespace: str = "default") -> str:
    """Generate cache key from prompt and namespace.

    Args:
        prompt: Prompt text
        namespace: Cache namespace

    Returns:
        Hash-based cache key
    """
    combined = f"{namespace}:{prompt}"
    return hashlib.sha256(combined.encode()).hexdigest()


def estimate_tokens(text: str) -> int:
    """Estimate token count for text (rough approximation).

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    # Rough approximation: ~4 chars per token
    return len(text) // 4


def serialize_response(response: Any) -> str:
    """Serialize response for storage.

    Args:
        response: Response object (string, dict, etc.)

    Returns:
        Serialized JSON string
    """
    import json

    return json.dumps(response)


def deserialize_response(data: str) -> Any:
    """Deserialize response from storage.

    Args:
        data: Serialized JSON string

    Returns:
        Deserialized response object
    """
    import json

    return json.loads(data)


__all__ = [
    "normalize_prompt",
    "hash_prompt",
    "estimate_tokens",
    "serialize_response",
    "deserialize_response",
]
