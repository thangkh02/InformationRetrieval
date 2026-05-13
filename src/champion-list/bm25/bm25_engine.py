from __future__ import annotations

import heapq
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import joblib

CURRENT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = CURRENT_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from documents import NewsDocument, SearchResult
from text_utils import tokenize_underthesea_text


class BM25SearchEngine:
    def __init__(self, k1: float = 1.5, b: float = 0.75, champion_size: int = 8000) -> None:
        self.k1 = k1
        self.b = b
        self.champion_size = champion_size
        self.documents: list[NewsDocument] = []
        self.doc_lengths: list[int] = []
        self.avgdl: float = 0.0
        self.inverted_index: dict[str, list[tuple[int, int]]] = {}
        self.champion_index: dict[str, list[tuple[int, int]]] = {}
        self.idf: dict[str, float] = {}

    def fit(self, documents: list[NewsDocument]) -> None:
        if not documents:
            raise ValueError("No documents provided for indexing.")

        self.documents = documents
        self.doc_lengths = []
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)

        for doc_idx, doc in enumerate(documents):
            tokens = list(tokenize_underthesea_text(doc.text))
            self.doc_lengths.append(len(tokens))
            if not tokens:
                continue

            term_freqs = Counter(tokens)
            for term, tf in term_freqs.items():
                postings[term].append((doc_idx, tf))

        self.avgdl = float(sum(self.doc_lengths) / len(self.doc_lengths)) if self.doc_lengths else 0.0
        self.inverted_index = dict(postings)
        self.idf = {}

        total_docs = len(documents)
        for term, posting_list in self.inverted_index.items():
            df = len(posting_list)
            self.idf[term] = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))

        self.champion_index = {}
        if self.champion_size > 0:
            for term, posting_list in self.inverted_index.items():
                idf = self.idf.get(term, 0.0)
                scored_postings = []
                for doc_idx, tf in posting_list:
                    dl = self.doc_lengths[doc_idx]
                    denom_norm = self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl)) if self.avgdl > 0 else self.k1
                    numer = tf * (self.k1 + 1.0)
                    denom = tf + denom_norm
                    score = idf * (numer / denom)
                    scored_postings.append((score, doc_idx, tf))

                top_postings = heapq.nlargest(self.champion_size, scored_postings, key=lambda item: item[0])
                self.champion_index[term] = [(doc_idx, tf) for _, doc_idx, tf in top_postings]

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        if not self.documents:
            raise RuntimeError("The engine has not been fitted yet.")
        if not query.strip():
            return []

        query_terms = list(tokenize_underthesea_text(query))
        if not query_terms:
            return []

        candidate_scores: dict[int, float] = defaultdict(float)
        query_term_set = set(query_terms)

        for term in query_term_set:
            posting_list = self.champion_index.get(term) if self.champion_size > 0 else self.inverted_index.get(term)
            if not posting_list:
                continue
            idf = self.idf.get(term, 0.0)
            for doc_idx, tf in posting_list:
                dl = self.doc_lengths[doc_idx]
                denom_norm = self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl)) if self.avgdl > 0 else self.k1
                numer = tf * (self.k1 + 1.0)
                denom = tf + denom_norm
                candidate_scores[doc_idx] += idf * (numer / denom)

        if not candidate_scores:
            return []

        top_items = heapq.nlargest(min(top_k, len(candidate_scores)), candidate_scores.items(), key=lambda item: item[1])

        results: list[SearchResult] = []
        for doc_idx, score in top_items:
            doc = self.documents[doc_idx]
            results.append(
                SearchResult(
                    doc_id=doc.doc_id,
                    score=float(score),
                    title=doc.title,
                    summary=doc.summary,
                    category=doc.category,
                    content=doc.content,
                )
            )
        return results

    def save(self, model_dir: str | Path) -> None:
        if not self.documents:
            raise RuntimeError("Nothing to save. Fit the engine first.")
        path = Path(model_dir)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "k1": self.k1,
                "b": self.b,
                "champion_size": self.champion_size,
                "documents": self.documents,
                "doc_lengths": self.doc_lengths,
                "avgdl": self.avgdl,
                "inverted_index": self.inverted_index,
                "champion_index": self.champion_index,
                "idf": self.idf,
            },
            path / "bm25_model.joblib",
        )

    @classmethod
    def load(cls, model_dir: str | Path) -> "BM25SearchEngine":
        payload = joblib.load(Path(model_dir) / "bm25_model.joblib")
        engine = cls(k1=payload["k1"], b=payload["b"], champion_size=payload.get("champion_size", 8000))
        engine.documents = payload["documents"]
        engine.doc_lengths = payload["doc_lengths"]
        engine.avgdl = payload["avgdl"]
        engine.inverted_index = payload["inverted_index"]
        engine.champion_index = payload.get("champion_index", {})
        engine.idf = payload["idf"]
        return engine
