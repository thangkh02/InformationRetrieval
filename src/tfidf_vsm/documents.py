from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ZaloDocument:
    doc_id: str
    title: str
    content: str

    @property
    def text(self) -> str:
        parts = [self.title, self.content]
        return " ".join(part.strip() for part in parts if part and part.strip())


@dataclass(slots=True)
class ZaloQuery:
    query_id: str
    text: str


@dataclass(slots=True)
class ZaloResult:
    doc_id: str
    score: float
    title: str
    content: str
