from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class LegalDocument:
    doc_id: str | int
    title: str
    content: str

    @property
    def text(self) -> str:
        parts = [self.title, self.content]
        return " ".join(part.strip() for part in parts if part and part.strip())


@dataclass(slots=True)
class SearchResult:
    doc_id: str | int
    score: float
    title: str
    content: str
