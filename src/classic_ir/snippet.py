from __future__ import annotations

import html
import re

from .preprocess import normalize_text, query_terms


def make_snippet(text: str, query: str, window: int = 40, highlight: bool = True) -> str:
    words = text.split()
    if not words:
        return ""

    normalized_terms = set(query_terms(query))
    first_match = 0
    for idx, word in enumerate(words):
        if normalize_text(word) in normalized_terms:
            first_match = idx
            break

    start = max(0, first_match - window)
    end = min(len(words), first_match + window + 1)
    snippet = " ".join(words[start:end])
    if start > 0:
        snippet = "... " + snippet
    if end < len(words):
        snippet += " ..."

    if not highlight:
        return snippet
    return highlight_text(snippet, query)


def highlight_text(text: str, query: str) -> str:
    terms = sorted(set(query_terms(query)), key=len, reverse=True)
    raw_terms = [term.replace("_", " ") for term in terms]
    raw_terms = [term for term in raw_terms if term]
    if not raw_terms:
        return html.escape(text)

    pattern = re.compile("|".join(re.escape(term) for term in raw_terms), flags=re.IGNORECASE)
    parts: list[str] = []
    last = 0
    for match in pattern.finditer(text):
        parts.append(html.escape(text[last : match.start()]))
        parts.append(f"<mark>{html.escape(match.group(0))}</mark>")
        last = match.end()
    parts.append(html.escape(text[last:]))
    return "".join(parts)
