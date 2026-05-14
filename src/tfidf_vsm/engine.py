from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pickle

from .documents import Document, SearchResult


class TfidfVSMEngine:
    def __init__(self, max_features: int = 200_000, ngram_range: tuple[int, int] = (1, 2), min_df: int = 1) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df

        self.documents: list[Document] = []
        self.vocab: dict[str, int] = {}
        self.idf: list[float] = []
        self.doc_vectors: list[dict[int, float]] = []
        self.doc_norms: list[float] = []
        self.inverted_index: dict[int, list[tuple[int, float]]] = {}

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.lower()
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _extract_terms(self, text: str) -> list[str]:
        norm = self._normalize_text(text)
        if not norm:
            return []

        tokens = norm.split()
        terms: list[str] = []
        n_min, n_max = self.ngram_range
        for n in range(n_min, n_max + 1):
            if n <= 0 or len(tokens) < n:
                continue
            if n == 1:
                terms.extend(tokens)
            else:
                for i in range(len(tokens) - n + 1):
                    terms.append(" ".join(tokens[i : i + n]))
        return terms

    @staticmethod
    def _l2_norm(vec: dict[int, float]) -> float:
        return math.sqrt(sum(v * v for v in vec.values()))

    def fit(self, documents: list[Document]) -> None:
        if not documents:
            raise ValueError("No documents provided.")

        self.documents = documents
        doc_term_counts: list[Counter[str]] = []
        df_counter: Counter[str] = Counter()

        for doc in documents:
            counts = Counter(self._extract_terms(doc.text))
            doc_term_counts.append(counts)
            df_counter.update(counts.keys())

        filtered_terms = [term for term, df in df_counter.items() if df >= self.min_df]
        filtered_terms.sort(key=lambda t: (-df_counter[t], t))
        if self.max_features > 0:
            filtered_terms = filtered_terms[: self.max_features]

        self.vocab = {term: idx for idx, term in enumerate(filtered_terms)}

        doc_count = len(documents)
        self.idf = [0.0] * len(self.vocab)
        for term, idx in self.vocab.items():
            df = df_counter[term]
            self.idf[idx] = math.log((doc_count + 1.0) / (df + 1.0)) + 1.0

        self.doc_vectors = []
        self.doc_norms = []
        postings: dict[int, list[tuple[int, float]]] = defaultdict(list)

        for doc_idx, term_counts in enumerate(doc_term_counts):
            vec: dict[int, float] = {}
            for term, tf in term_counts.items():
                term_idx = self.vocab.get(term)
                if term_idx is None:
                    continue
                vec[term_idx] = (1.0 + math.log(tf)) * self.idf[term_idx]

            norm = self._l2_norm(vec)
            self.doc_vectors.append(vec)
            self.doc_norms.append(norm)

            if norm == 0.0:
                continue
            for term_idx, weight in vec.items():
                postings[term_idx].append((doc_idx, weight))

        self.inverted_index = dict(postings)

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        if not self.doc_vectors:
            raise RuntimeError("Engine is not fitted.")
        if not query.strip():
            return []

        q_counts = Counter(self._extract_terms(query))
        if not q_counts:
            return []

        q_vec: dict[int, float] = {}
        for term, tf in q_counts.items():
            term_idx = self.vocab.get(term)
            if term_idx is None:
                continue
            q_vec[term_idx] = (1.0 + math.log(tf)) * self.idf[term_idx]

        if not q_vec:
            return []

        q_norm = self._l2_norm(q_vec)
        if q_norm == 0.0:
            return []

        dot_scores: dict[int, float] = defaultdict(float)
        for term_idx, q_weight in q_vec.items():
            for doc_idx, d_weight in self.inverted_index.get(term_idx, []):
                dot_scores[doc_idx] += q_weight * d_weight

        scored: list[tuple[int, float]] = []
        for doc_idx, dot in dot_scores.items():
            d_norm = self.doc_norms[doc_idx]
            if d_norm == 0.0:
                continue
            score = dot / (q_norm * d_norm)
            if score > 0.0:
                scored.append((doc_idx, score))

        if not scored:
            return []

        scored.sort(key=lambda x: x[1], reverse=True)
        top_hits = scored[: min(top_k, len(scored))]

        return [
            SearchResult(
                doc_id=self.documents[doc_idx].doc_id,
                score=float(score),
                title=self.documents[doc_idx].title,
                content=self.documents[doc_idx].content,
            )
            for doc_idx, score in top_hits
        ]

    def save(self, model_dir: str | Path) -> None:
        if not self.doc_vectors:
            raise RuntimeError("Nothing to save.")

        out = Path(model_dir)
        out.mkdir(parents=True, exist_ok=True)
        payload = {
            "max_features": self.max_features,
            "ngram_range": self.ngram_range,
            "min_df": self.min_df,
            "documents": self.documents,
            "vocab": self.vocab,
            "idf": self.idf,
            "doc_vectors": self.doc_vectors,
            "doc_norms": self.doc_norms,
            "inverted_index": self.inverted_index,
        }
        with open(out / "tfidf_vsm.pkl", "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, model_dir: str | Path) -> "TfidfVSMEngine":
        with open(Path(model_dir) / "tfidf_vsm.pkl", "rb") as f:
            payload = pickle.load(f)
        engine = cls(
            max_features=payload.get("max_features", 200_000),
            ngram_range=tuple(payload.get("ngram_range", (1, 2))),
            min_df=payload.get("min_df", 1),
        )
        engine.documents = payload["documents"]
        engine.vocab = payload["vocab"]
        engine.idf = payload["idf"]
        engine.doc_vectors = payload["doc_vectors"]
        engine.doc_norms = payload["doc_norms"]
        engine.inverted_index = payload["inverted_index"]
        return engine

