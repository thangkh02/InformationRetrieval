from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class NewsDocument:
    doc_id: int
    title: str
    summary: str
    category: str
    content: str

    @property
    def text(self) -> str:
        parts = [self.title, self.summary, self.category, self.content]
        return " ".join(part.strip() for part in parts if part and part.strip())


@dataclass(slots=True)
class SearchResult:
    doc_id: int
    score: float
    title: str
    summary: str
    category: str
    content: str

