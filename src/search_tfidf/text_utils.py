from __future__ import annotations

import re

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
        from underthesea import word_tokenize

        text = word_tokenize(text, format="text")
    except Exception:
        pass

    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text
