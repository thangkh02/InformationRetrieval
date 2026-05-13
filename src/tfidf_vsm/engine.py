from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .documents import ZaloDocument, ZaloResult


class ZaloTfidfVSMEngine:
    def __init__(self, max_features: int = 200_000, ngram_range: tuple[int, int] = (1, 2), min_df: int = 1) -> None:
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
        )
        self.documents: list[ZaloDocument] = []
        self.matrix = None

    def fit(self, documents: list[ZaloDocument]) -> None:
        if not documents:
            raise ValueError("No documents provided.")
        self.documents = documents
        self.matrix = self.vectorizer.fit_transform([doc.text for doc in documents])

    def search(self, query: str, top_k: int = 10) -> list[ZaloResult]:
        if self.matrix is None:
            raise RuntimeError("Engine is not fitted.")
        if not query.strip():
            return []

        q_vec = self.vectorizer.transform([query])
        if q_vec.nnz == 0:
            return []

        scores = cosine_similarity(q_vec, self.matrix).ravel()
        top_k = min(top_k, len(self.documents))
        top_idx = np.argpartition(-scores, top_k - 1)[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        return [
            ZaloResult(
                doc_id=self.documents[i].doc_id,
                score=float(scores[i]),
                title=self.documents[i].title,
                content=self.documents[i].content,
            )
            for i in top_idx
        ]

    def save(self, model_dir: str | Path) -> None:
        if self.matrix is None:
            raise RuntimeError("Nothing to save.")
        out = Path(model_dir)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "vectorizer": self.vectorizer,
                "documents": self.documents,
                "matrix": self.matrix,
            },
            out / "zalo_tfidf_vsm.joblib",
        )

    @classmethod
    def load(cls, model_dir: str | Path) -> "ZaloTfidfVSMEngine":
        payload = joblib.load(Path(model_dir) / "zalo_tfidf_vsm.joblib")
        engine = cls()
        engine.vectorizer = payload["vectorizer"]
        engine.documents = payload["documents"]
        engine.matrix = payload["matrix"]
        return engine
