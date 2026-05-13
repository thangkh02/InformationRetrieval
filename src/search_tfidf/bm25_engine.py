from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from rank_bm25 import BM25Okapi

from .documents import NewsDocument, SearchResult
from .text_utils import tokenize_text


class BM25SearchEngine:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.documents: list[NewsDocument] = []
        self.tokenized_corpus: list[list[str]] = []
        self.bm25: BM25Okapi | None = None

    def fit(self, documents: list[NewsDocument]) -> None:
        if not documents:
            raise ValueError("No documents provided for indexing.")
        self.documents = documents
        self.tokenized_corpus = [tokenize_text(doc.text) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        if self.bm25 is None:
            raise RuntimeError("The engine has not been fitted yet.")
        if not query.strip():
            return []

        tokens = tokenize_text(query)
        if not tokens:
            return []

        scores = np.asarray(self.bm25.get_scores(tokens), dtype=np.float32)
        if scores.size == 0:
            return []

        top_k = min(top_k, scores.size)
        top_indices = np.argpartition(-scores, top_k - 1)[:top_k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        results: list[SearchResult] = []
        for idx in top_indices:
            doc = self.documents[int(idx)]
            results.append(
                SearchResult(
                    doc_id=doc.doc_id,
                    score=float(scores[idx]),
                    title=doc.title,
                    summary=doc.summary,
                    category=doc.category,
                    content=doc.content,
                )
            )
        return results

    def save(self, model_dir: str | Path) -> None:
        if self.bm25 is None:
            raise RuntimeError("Nothing to save. Fit the engine first.")
        path = Path(model_dir)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "k1": self.k1,
                "b": self.b,
                "documents": self.documents,
                "tokenized_corpus": self.tokenized_corpus,
            },
            path / "bm25_model.joblib",
        )

    @classmethod
    def load(cls, model_dir: str | Path) -> "BM25SearchEngine":
        payload = joblib.load(Path(model_dir) / "bm25_model.joblib")
        engine = cls(k1=payload["k1"], b=payload["b"])
        engine.documents = payload["documents"]
        engine.tokenized_corpus = payload["tokenized_corpus"]
        engine.bm25 = BM25Okapi(engine.tokenized_corpus, k1=engine.k1, b=engine.b)
        return engine
