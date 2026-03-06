# semantic-llm-cache

**Async semantic caching for LLM API calls — reduce costs with one decorator.**

> **Fork of [karthyick/prompt-cache](https://github.com/karthyick/prompt-cache)** — fully converted to async for use with async frameworks (FastAPI, aiohttp, Starlette, etc.).

## Overview

LLM API calls are expensive and slow. In production applications, **20-40% of prompts are semantically identical** but get charged as separate API calls. `semantic-llm-cache` solves this with a simple decorator that:

- ✅ **Caches semantically similar prompts** (not just exact matches)
- ✅ **Reduces API costs by 20-40%**
- ✅ **Returns cached responses in <10ms**
- ✅ **Works with any LLM provider** (OpenAI, Anthropic, Ollama, local models)
- ✅ **Fully async** — native `async/await` throughout, no event loop blocking
- ✅ **Auto-detects** sync vs async decorated functions — one decorator for both

## What changed from the original


| Area                 | Original                  | This fork                                                           |
| ---------------------- | --------------------------- | --------------------------------------------------------------------- |
| Backends             | sync (`sqlite3`, `redis`) | async (`aiosqlite`, `redis.asyncio`)                                |
| `@cache` decorator   | sync only                 | auto-detects async/sync                                             |
| `EmbeddingCache`     | sync`encode()`            | adds`async aencode()` via `asyncio.to_thread`                       |
| `CacheContext`       | sync only                 | supports both`with` and `async with`                                |
| `CachedLLM`          | `chat()`                  | adds`achat()`                                                       |
| Utility functions    | sync                      | `clear_cache`, `invalidate`, `warm_cache`, `export_cache` all async |
| `StorageBackend` ABC | sync abstract methods     | all abstract methods are`async def`                                 |
| Min Python           | 3.9                       | 3.10 (uses`X                                                        |

## Installation

Not yet published to PyPI. Install directly from the repository:

```bash
# Clone
git clone https://github.com/nomyo-ai/async-semantic-llm-cache.git
cd prompt-cache

# Core (exact match only, SQLite backend)
pip install .

# With semantic similarity (sentence-transformers)
pip install ".[semantic]"

# With Redis backend
pip install ".[redis]"

# Everything
pip install ".[all]"
```

Or install directly via pip from git:

```bash
# Core only
pip install "git+https://github.com/nomyo-ai/async-semantic-llm-cache.git"

# With extras — use PEP 508 @ syntax (extras after the URL don't work with pip)
pip install "semantic-llm-cache[semantic] @ git+https://github.com/nomyo-ai/async-semantic-llm-cache.git"
pip install "semantic-llm-cache[redis] @ git+https://github.com/nomyo-ai/async-semantic-llm-cache.git"
pip install "semantic-llm-cache[all] @ git+https://github.com/nomyo-ai/async-semantic-llm-cache.git"
```

## Quick Start

### Async function (FastAPI, aiohttp, etc.)

```python
from semantic_llm_cache import cache

@cache(similarity=0.95, ttl=3600)
async def ask_llm(prompt: str) -> str:
    return await call_ollama(prompt)

# First call — LLM hit
await ask_llm("What is Python?")

# Second call — cache hit (<10ms, free)
await ask_llm("What's Python?")  # 95% similar → cache hit
```

### Sync function (backwards compatible)

```python
from semantic_llm_cache import cache

@cache()
def ask_llm_sync(prompt: str) -> str:
    return call_openai(prompt)  # works, but don't use inside a running event loop
```

### Semantic Matching

```python
from semantic_llm_cache import cache

@cache(similarity=0.90)
async def ask_llm(prompt: str) -> str:
    return await call_ollama(prompt)

await ask_llm("What is Python?")   # LLM call
await ask_llm("What's Python?")    # cache hit (95% similar)
await ask_llm("Explain Python")    # cache hit (91% similar)
await ask_llm("What is Rust?")     # LLM call (different topic)
```

### SQLite backend (default, persistent)

```python
from semantic_llm_cache import cache
from semantic_llm_cache.backends import SQLiteBackend

backend = SQLiteBackend(db_path="my_cache.db")

@cache(backend=backend, similarity=0.95)
async def ask_llm(prompt: str) -> str:
    return await call_ollama(prompt)
```

### Redis backend (distributed)

```python
from semantic_llm_cache import cache
from semantic_llm_cache.backends import RedisBackend

backend = RedisBackend(url="redis://localhost:6379/0")
await backend.ping()  # verify connection (replaces __init__ connection test)

@cache(backend=backend, similarity=0.95)
async def ask_llm(prompt: str) -> str:
    return await call_ollama(prompt)
```

### Cache Statistics

```python
from semantic_llm_cache import get_stats

stats = get_stats()
# {
#     "hits": 1547,
#     "misses": 892,
#     "hit_rate": 0.634,
#     "estimated_savings_usd": 3.09,
#     "total_saved_ms": 773500
# }
```

### Cache Management

```python
from semantic_llm_cache.stats import clear_cache, invalidate

# Clear all cached entries
await clear_cache()

# Invalidate entries matching a pattern
await invalidate(pattern="Python")
```

### Async context manager

```python
from semantic_llm_cache import CacheContext

async with CacheContext(similarity=0.9) as ctx:
    result1 = await any_cached_llm_call("prompt 1")
    result2 = await any_cached_llm_call("prompt 2")

print(ctx.stats)  # {"hits": 1, "misses": 1}
```

### CachedLLM wrapper

```python
from semantic_llm_cache import CachedLLM

llm = CachedLLM(similarity=0.9, ttl=3600)
response = await llm.achat("What is Python?", llm_func=my_async_llm)
```

## API Reference

### `@cache()` Decorator

```python
@cache(
    similarity: float = 1.0,      # 1.0 = exact match, 0.9 = semantic
    ttl: int = 3600,              # seconds, None = forever
    backend: Backend = None,      # None = in-memory
    namespace: str = "default",   # isolate different use cases
    enabled: bool = True,         # toggle for debugging
    key_func: Callable = None,    # custom cache key
)
async def my_llm_function(prompt: str) -> str:
    ...
```

### Parameters


| Parameter    | Type       | Default     | Description                                               |
| -------------- | ------------ | ------------- | ----------------------------------------------------------- |
| `similarity` | `float`    | `1.0`       | Cosine similarity threshold (1.0 = exact, 0.9 = semantic) |
| `ttl`        | `int       | None`       | `3600`                                                    |
| `backend`    | `Backend`  | `None`      | Storage backend (None = in-memory)                        |
| `namespace`  | `str`      | `"default"` | Isolate different use cases                               |
| `enabled`    | `bool`     | `True`      | Enable/disable caching                                    |
| `key_func`   | `Callable` | `None`      | Custom cache key function                                 |

### Utility Functions

```python
from semantic_llm_cache import get_stats          # sync — safe anywhere
from semantic_llm_cache.stats import (
    clear_cache,   # async
    invalidate,    # async
    warm_cache,    # async
    export_cache,  # async
)
```

## Backends


| Backend         | Description                          | I/O                        |
| ----------------- | -------------------------------------- | ---------------------------- |
| `MemoryBackend` | In-memory LRU (default)              | none — runs in event loop |
| `SQLiteBackend` | Persistent, file-based (`aiosqlite`) | async non-blocking         |
| `RedisBackend`  | Distributed (`redis.asyncio`)        | async non-blocking         |

## Embedding Providers


| Provider                      | Quality                      | Notes                      |
| ------------------------------- | ------------------------------ | ---------------------------- |
| `DummyEmbeddingProvider`      | hash-only, no semantic match | zero deps, default         |
| `SentenceTransformerProvider` | high (local model)           | requires`[semantic]` extra |
| `OpenAIEmbeddingProvider`     | high (API)                   | requires`[openai]` extra   |

Embedding inference is offloaded via `asyncio.to_thread` — model loading is blocking and should be done at application startup, not on first request.

```python
from semantic_llm_cache.similarity import create_embedding_provider, EmbeddingCache

# Pre-load at startup (blocking — do this in lifespan, not a request handler)
provider = create_embedding_provider("sentence-transformer")
embedding_cache = EmbeddingCache(provider=provider)

# Use in request handlers (non-blocking)
embedding = await embedding_cache.aencode("my prompt")
```

## Performance


| Metric                     | Value                                    |
| ---------------------------- | ------------------------------------------ |
| Cache hit latency          | <10ms                                    |
| Embedding overhead on miss | ~50ms (sentence-transformers, offloaded) |
| Typical hit rate           | 25-40%                                   |
| Cost reduction             | 20-40%                                   |

## Requirements

- Python >= 3.10
- numpy >= 1.24.0
- aiosqlite >= 0.19.0

### Optional

- `sentence-transformers >= 2.2.0` — semantic matching
- `redis >= 4.2.0` — Redis backend (includes `redis.asyncio`)
- `openai >= 1.0.0` — OpenAI embeddings

## License

MIT — see [LICENSE](LICENSE).

## Credits

Original library by **Karthick Raja M** ([@karthyick](https://github.com/karthyick)).
Async conversion by this fork.
