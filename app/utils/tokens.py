from __future__ import annotations
try:
    import tiktoken
except ImportError:  # pragma: no cover
    tiktoken = None


def count_tokens(text: str) -> int:
    if tiktoken is None:
        return max(1, len(text) // 4)
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))
