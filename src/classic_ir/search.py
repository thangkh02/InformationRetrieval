from __future__ import annotations

from dataclasses import dataclass
import math
import re

from .index import ClassicIndex
from .preprocess import query_terms
from .snippet import make_snippet


@dataclass(slots=True)
class SearchResult:
    rank: int
    doc_id: str
    score: float
    title: str
    snippet: str


class ClassicSearchEngine:
    def __init__(self, index: ClassicIndex, k1: float = 1.5, b: float = 0.75) -> None:
        self.index = index
        self.k1 = k1
        self.b = b
        self.all_doc_ids = set(index.doc_store)

    @classmethod
    def load(cls, index_dir: str, k1: float = 1.5, b: float = 0.75) -> "ClassicSearchEngine":
        return cls(ClassicIndex.load(index_dir), k1=k1, b=b)

    @classmethod
    def load_from_bm25_model(
        cls,
        model_dir: str,
        tokenized_corpus_path: str | None = None,
        positional_index_path: str | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> "ClassicSearchEngine":
        index = ClassicIndex.from_bm25_model(
            model_dir,
            tokenized_corpus_path=tokenized_corpus_path,
            positional_index_path=positional_index_path,
        )
        return cls(index, k1=k1, b=b)

    def bm25_search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        terms = query_terms(query)
        if not terms:
            return []

        score_by_doc: dict[str, float] = {}
        for term in terms:
            for doc_id in self.index.inverted_index.get(term, {}):
                score_by_doc[doc_id] = score_by_doc.get(doc_id, 0.0) + self._bm25_term_score(term, doc_id)

        scored: list[tuple[str, float]] = [(doc_id, score) for doc_id, score in score_by_doc.items() if score > 0]

        scored.sort(key=lambda item: (-item[1], item[0]))
        return self._format_results(scored[:top_k], query)

    def fielded_bm25_search(
        self,
        query: str,
        top_k: int = 10,
        title_weight: float = 0.2,
        text_weight: float = 1.0,
    ) -> list[SearchResult]:
        terms = query_terms(query)
        if not terms:
            return []

        score_by_doc: dict[str, float] = {}
        for term in terms:
            for doc_id in self.index.inverted_index.get(term, {}):
                score_by_doc[doc_id] = score_by_doc.get(doc_id, 0.0) + text_weight * self._bm25_term_score(term, doc_id)
            for doc_id in self.index.title_inverted_index.get(term, {}):
                score_by_doc[doc_id] = score_by_doc.get(doc_id, 0.0) + title_weight * self._title_bm25_term_score(term, doc_id)

        scored = [(doc_id, score) for doc_id, score in score_by_doc.items() if score > 0]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return self._format_results(scored[:top_k], query)

    def boolean_search(self, expression: str, top_k: int = 10) -> list[SearchResult]:
        doc_ids = self.boolean_doc_ids(expression)
        return self._format_doc_ids(sorted(doc_ids)[:top_k], expression)

    def boolean_search_ranked(self, expression: str, top_k: int = 10) -> list[SearchResult]:
        doc_ids = self.boolean_doc_ids(expression)
        terms = [term for term in query_terms(expression) if term not in {"and", "or", "not"}]
        scored = []
        for doc_id in doc_ids:
            overlap = sum(1 for term in terms if doc_id in self.index.inverted_index.get(term, {}))
            scored.append((doc_id, float(overlap)))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return self._format_results(scored[:top_k], expression)

    def boolean_doc_ids(self, expression: str) -> set[str]:
        parts = re.split(r"\s+(AND|OR|NOT)\s+", expression, flags=re.IGNORECASE)
        if not parts:
            return set()

        current: set[str] | None = None
        op = "OR"
        negate_next = False
        for raw in parts:
            token = raw.strip()
            if not token:
                continue
            upper = token.upper()
            if upper in {"AND", "OR"}:
                op = upper
                continue
            if upper == "NOT":
                negate_next = True
                continue

            token_docs = self._docs_for_text(token)
            if negate_next:
                token_docs = self.all_doc_ids - token_docs
                negate_next = False

            if current is None:
                current = set(token_docs)
            elif op == "AND":
                current &= token_docs
            else:
                current |= token_docs

        return current or set()

    def phrase_search(self, phrase: str, top_k: int = 10) -> list[SearchResult]:
        terms = query_terms(phrase.strip('"'))
        if not terms:
            return []

        matches = self.phrase_doc_ids(terms)
        return self._format_doc_ids(sorted(matches)[:top_k], phrase)

    def phrase_search_ranked(self, phrase: str, top_k: int = 10) -> list[SearchResult]:
        terms = query_terms(phrase.strip('"'))
        if not terms:
            return []

        matches = self.phrase_doc_ids(terms)
        scored = [(doc_id, sum(self._bm25_term_score(term, doc_id) for term in terms)) for doc_id in matches]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return self._format_results(scored[:top_k], phrase)

    def near_search(self, left: str, right: str, distance: int = 5, top_k: int = 10) -> list[SearchResult]:
        left_terms = query_terms(left)
        right_terms = query_terms(right)
        if len(left_terms) != 1 or len(right_terms) != 1 or distance < 1:
            return []

        left_term = left_terms[0]
        right_term = right_terms[0]
        matches = self.near_doc_ids(left_term, right_term, distance)
        scored = [
            (
                doc_id,
                self._bm25_term_score(left_term, doc_id) + self._bm25_term_score(right_term, doc_id),
            )
            for doc_id in matches
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return self._format_results(scored[:top_k], f"{left} {right}")

    def near_doc_ids(self, left_term: str, right_term: str, distance: int) -> set[str]:
        left_postings = self.index.positional_index.get(left_term, {})
        right_postings = self.index.positional_index.get(right_term, {})
        candidates = set(left_postings) & set(right_postings)
        return {
            doc_id
            for doc_id in candidates
            if self._has_positions_within_distance(left_postings[doc_id], right_postings[doc_id], distance)
        }

    def phrase_doc_ids(self, terms: list[str]) -> set[str]:
        candidates = self._candidate_docs_for_all_terms(terms)
        return {doc_id for doc_id in candidates if self.contains_phrase(doc_id, terms)}

    def contains_phrase(self, doc_id: str, terms: list[str]) -> bool:
        if not terms:
            return False
        first_positions = self.index.positional_index.get(terms[0], {}).get(doc_id, [])
        if not first_positions:
            return False

        remaining = [set(self.index.positional_index.get(term, {}).get(doc_id, [])) for term in terms[1:]]
        for start in first_positions:
            if all((start + offset + 1) in positions for offset, positions in enumerate(remaining)):
                return True
        return False

    def _bm25_term_score(self, term: str, doc_id: str) -> float:
        tf = self.index.inverted_index.get(term, {}).get(doc_id, 0)
        if tf <= 0:
            return 0.0
        df = self.index.df.get(term, 0)
        if df <= 0:
            return 0.0
        idf = math.log(1 + (self.index.n_docs - df + 0.5) / (df + 0.5))
        dl = self.index.doc_len.get(doc_id, 0)
        denom = tf + self.k1 * (1 - self.b + self.b * dl / max(self.index.avgdl, 1e-9))
        return idf * (tf * (self.k1 + 1)) / denom

    def _title_bm25_term_score(self, term: str, doc_id: str) -> float:
        tf = self.index.title_inverted_index.get(term, {}).get(doc_id, 0)
        if tf <= 0:
            return 0.0
        df = self.index.title_df.get(term, 0)
        if df <= 0:
            return 0.0
        idf = math.log(1 + (self.index.n_docs - df + 0.5) / (df + 0.5))
        dl = self.index.title_doc_len.get(doc_id, 0)
        denom = tf + self.k1 * (1 - self.b + self.b * dl / max(self.index.title_avgdl, 1e-9))
        return idf * (tf * (self.k1 + 1)) / denom

    def _docs_for_text(self, text: str) -> set[str]:
        terms = query_terms(text)
        docs: set[str] = set()
        for term in terms:
            docs.update(self.index.inverted_index.get(term, {}))
        return docs

    def _candidate_docs_for_all_terms(self, terms: list[str]) -> set[str]:
        postings = [set(self.index.inverted_index.get(term, {})) for term in terms]
        if not postings:
            return set()
        candidates = postings[0]
        for posting in postings[1:]:
            candidates &= posting
        return candidates

    @staticmethod
    def _has_positions_within_distance(left_positions: list[int], right_positions: list[int], distance: int) -> bool:
        i = 0
        j = 0
        while i < len(left_positions) and j < len(right_positions):
            gap = abs(left_positions[i] - right_positions[j])
            if 0 < gap <= distance:
                return True
            if left_positions[i] < right_positions[j]:
                i += 1
            else:
                j += 1
        return False

    def _format_results(self, scored: list[tuple[str, float]], query: str) -> list[SearchResult]:
        results: list[SearchResult] = []
        for rank, (doc_id, score) in enumerate(scored, start=1):
            doc = self.index.doc_store[doc_id]
            text = " ".join(part for part in [doc.get("title", ""), doc.get("text", "")] if part)
            results.append(
                SearchResult(
                    rank=rank,
                    doc_id=doc_id,
                    score=float(score),
                    title=doc.get("title", ""),
                    snippet=make_snippet(text, query),
                )
            )
        return results

    def _format_doc_ids(self, doc_ids: list[str], query: str) -> list[SearchResult]:
        return self._format_results([(doc_id, 1.0) for doc_id in doc_ids], query)
