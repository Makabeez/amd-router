"""Token counting utilities for cost estimation."""

from __future__ import annotations

import functools


@functools.lru_cache(maxsize=8)
def _get_encoder(name: str = "cl100k_base"):
    try:
        import tiktoken

        return tiktoken.get_encoding(name)
    except (ImportError, Exception):
        return None


def count_tokens(text: str, encoding: str = "cl100k_base") -> int:
    """Approximate token count. Use for cost estimation, not scoring.

    Final scoring uses backend-reported usage from Fireworks.
    """
    enc = _get_encoder(encoding)
    if enc is None:
        # Crude fallback: ~4 chars per token
        return max(1, len(text) // 4)
    return len(enc.encode(text))
