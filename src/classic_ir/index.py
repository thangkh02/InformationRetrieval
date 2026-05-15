from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import pickle
import sys

import joblib

from .preprocess import tokenize


CHAMPION_ROOT = Path(__file__).resolve().parents[1] / "champion-list"
if str(CHAMPION_ROOT) not in sys.path:
    sys.path.insert(0, str(CHAMPION_ROOT))


@dataclass(slots=True)
class ClassicIndex:
    inverted_index: dict[str, dict[str, int]]
    positional_index: dict[str, dict[str, list[int]]]
    title_inverted_index: dict[str, dict[str, int]]
    title_df: dict[str, int]
    title_doc_len: dict[str, int]
    title_avgdl: float
    df: dict[str, int]
    doc_len: dict[str, int]
    doc_store: dict[str, dict[str, str]]
    avgdl: float
    n_docs: int

    @classmethod
    def from_bm25_model(
        cls,
        model_dir: str | Path,
        tokenized_corpus_path: str | Path | None = None,
        positional_index_path: str | Path | None = None,
    ) -> "ClassicIndex":
        payload = _load_bm25_payload(model_dir)
        doc_ids = [str(doc.doc_id) for doc in payload["documents"]]
        positional = _load_or_build_positional_index(positional_index_path, tokenized_corpus_path)

        inverted = _posting_lists_to_doc_ids(payload["inverted_index"], doc_ids)
        title_inverted, title_df, title_doc_len, title_avgdl = _build_title_index(payload["documents"])
        doc_store = {
            str(doc.doc_id): {"doc_id": str(doc.doc_id), "title": doc.title, "text": doc.content}
            for doc in payload["documents"]
        }

        return cls(
            inverted_index=inverted,
            positional_index=positional,
            title_inverted_index=title_inverted,
            title_df=title_df,
            title_doc_len=title_doc_len,
            title_avgdl=title_avgdl,
            df={term: len(postings) for term, postings in inverted.items()},
            doc_len=dict(zip(doc_ids, payload["doc_lengths"], strict=True)),
            doc_store=doc_store,
            avgdl=float(payload["avgdl"]),
            n_docs=len(doc_ids),
        )

    def save(self, output_dir: str | Path) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        _dump(self.inverted_index, path / "inverted_index.pkl")
        _dump(self.positional_index, path / "positional_index.pkl")
        _dump(self.title_inverted_index, path / "title_inverted_index.pkl")
        _dump(self.title_df, path / "title_df.pkl")
        _dump(self.title_doc_len, path / "title_doc_len.pkl")
        _dump(self.df, path / "df.pkl")
        _dump(self.doc_len, path / "doc_len.pkl")
        _dump(self.doc_store, path / "doc_store.pkl")
        _dump({"avgdl": self.avgdl, "N": self.n_docs, "title_avgdl": self.title_avgdl}, path / "meta.pkl")

    @classmethod
    def load(cls, index_dir: str | Path) -> "ClassicIndex":
        path = Path(index_dir)
        meta = _load(path / "meta.pkl")
        doc_store = _load(path / "doc_store.pkl")
        title_index_path = path / "title_inverted_index.pkl"
        if title_index_path.exists():
            title_inverted = _load(title_index_path)
            title_df = _load(path / "title_df.pkl")
            title_doc_len = _load(path / "title_doc_len.pkl")
            title_avgdl = float(meta.get("title_avgdl", 0.0))
        else:
            title_inverted, title_df, title_doc_len, title_avgdl = _build_title_index_from_store(doc_store)

        return cls(
            inverted_index=_load(path / "inverted_index.pkl"),
            positional_index=_load(path / "positional_index.pkl"),
            title_inverted_index=title_inverted,
            title_df=title_df,
            title_doc_len=title_doc_len,
            title_avgdl=title_avgdl,
            df=_load(path / "df.pkl"),
            doc_len=_load(path / "doc_len.pkl"),
            doc_store=doc_store,
            avgdl=float(meta["avgdl"]),
            n_docs=int(meta["N"]),
        )


def _load_bm25_payload(model_dir: str | Path) -> dict:
    return joblib.load(Path(model_dir) / "bm25_model.joblib")


def _posting_lists_to_doc_ids(
    inverted_index: dict[str, list[tuple[int, int]]],
    doc_ids: list[str],
) -> dict[str, dict[str, int]]:
    return {
        term: {doc_ids[doc_idx]: int(tf) for doc_idx, tf in posting_list}
        for term, posting_list in inverted_index.items()
    }


def _load_or_build_positional_index(
    positional_index_path: str | Path | None,
    tokenized_corpus_path: str | Path | None,
) -> dict[str, dict[str, list[int]]]:
    if positional_index_path and Path(positional_index_path).exists():
        return _load(Path(positional_index_path))
    if tokenized_corpus_path:
        return build_positional_index_from_tokens(tokenized_corpus_path)
    return {}


def build_positional_index_from_tokens(path: str | Path) -> dict[str, dict[str, list[int]]]:
    positional: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            doc_id = str(item.get("doc_id", item.get("id", item.get("_id"))))
            tokens = [str(token) for token in item.get("tokens", []) if str(token).strip()]
            for pos, token in enumerate(tokens):
                positional[token][doc_id].append(pos)
    return {term: dict(postings) for term, postings in positional.items()}


def _build_title_index(documents) -> tuple[dict[str, dict[str, int]], dict[str, int], dict[str, int], float]:
    title_inverted: dict[str, dict[str, int]] = defaultdict(dict)
    title_doc_len: dict[str, int] = {}
    for doc in documents:
        doc_id = str(doc.doc_id)
        tokens = tokenize(doc.title)
        title_doc_len[doc_id] = len(tokens)
        for term, tf in Counter(tokens).items():
            title_inverted[term][doc_id] = int(tf)

    frozen = {term: dict(postings) for term, postings in title_inverted.items()}
    n_docs = len(documents)
    avgdl = sum(title_doc_len.values()) / n_docs if n_docs else 0.0
    return frozen, {term: len(postings) for term, postings in frozen.items()}, title_doc_len, avgdl


def _build_title_index_from_store(
    doc_store: dict[str, dict[str, str]],
) -> tuple[dict[str, dict[str, int]], dict[str, int], dict[str, int], float]:
    title_inverted: dict[str, dict[str, int]] = defaultdict(dict)
    title_doc_len: dict[str, int] = {}
    for doc_id, doc in doc_store.items():
        tokens = tokenize(doc.get("title", ""))
        title_doc_len[doc_id] = len(tokens)
        for term, tf in Counter(tokens).items():
            title_inverted[term][doc_id] = int(tf)

    frozen = {term: dict(postings) for term, postings in title_inverted.items()}
    n_docs = len(doc_store)
    avgdl = sum(title_doc_len.values()) / n_docs if n_docs else 0.0
    return frozen, {term: len(postings) for term, postings in frozen.items()}, title_doc_len, avgdl


def _dump(obj: object, path: Path) -> None:
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)
