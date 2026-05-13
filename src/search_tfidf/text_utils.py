from __future__ import annotations

import re
from functools import lru_cache

_NON_ALNUM_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = text.lower()
    text = _NON_ALNUM_RE.sub(" ", text)
    text = text.replace("_", " ")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def tokenize_text(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    return normalized.split()


def prepare_phobert_text(text: str) -> str:
    text = text.lower().strip()
    text = _WHITESPACE_RE.sub(" ", text)

    try:
        from underthesea.pipeline.word_tokenize import word_tokenize

        text = word_tokenize(text, format="text")
    except Exception:
        pass

    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


@lru_cache(maxsize=8192)
def tokenize_underthesea_text(text: str) -> tuple[str, ...]:
    text = text.lower().strip()
    text = _WHITESPACE_RE.sub(" ", text)

    try:
        from underthesea.pipeline.word_tokenize import word_tokenize

        tokenized = word_tokenize(text, format="text")
    except Exception:
        tokenized = text

    tokenized = _WHITESPACE_RE.sub(" ", tokenized).strip()
    if not tokenized:
        return ()

    tokens = [tok for tok in tokenized.split() if tok]
    return tuple(tokens)
