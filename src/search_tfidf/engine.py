from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .documents import NewsDocument, SearchResult


class TfidfSearchEngine:
    def __init__(
        self,
        max_features: int = 50_000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
    ) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
        )
        self.documents: list[NewsDocument] = []
        self.matrix = None

    def fit(self, documents: list[NewsDocument]) -> None:
        if not documents:
            raise ValueError("No documents provided for indexing.")
        self.documents = documents
        corpus = [doc.text for doc in documents]
        self.matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        if self.matrix is None:
            raise RuntimeError("The engine has not been fitted yet.")
        if not query.strip():
            return []

        query_vector = self.vectorizer.transform([query])
        if query_vector.nnz == 0:
            return []

        scores = cosine_similarity(query_vector, self.matrix).ravel()
        positive_indices = np.flatnonzero(scores > 0)
        if positive_indices.size == 0:
            return []

        top_k = min(top_k, positive_indices.size)
        positive_scores = scores[positive_indices]
        top_local_indices = np.argpartition(-positive_scores, top_k - 1)[:top_k]
        top_indices = positive_indices[top_local_indices]
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
        if self.matrix is None:
            raise RuntimeError("Nothing to save. Fit the engine first.")
        path = Path(model_dir)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "vectorizer": self.vectorizer,
                "documents": self.documents,
                "matrix": self.matrix,
            },
            path / "tfidf_model.joblib",
        )

    @classmethod
    def load(cls, model_dir: str | Path) -> "TfidfSearchEngine":
        payload = joblib.load(Path(model_dir) / "tfidf_model.joblib")
        vectorizer = payload["vectorizer"]
        engine = cls(
            max_features=vectorizer.max_features if vectorizer.max_features is not None else 50_000,
            ngram_range=vectorizer.ngram_range,
            min_df=vectorizer.min_df if isinstance(vectorizer.min_df, int) else 2,
        )
        engine.vectorizer = payload["vectorizer"]
        engine.documents = payload["documents"]
        engine.matrix = payload["matrix"]
        return engine

