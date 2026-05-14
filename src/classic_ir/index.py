from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
import pickle

from .data_loader import Document
from .preprocess import tokenize


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
    def build(cls, documents: list[Document]) -> "ClassicIndex":
        inverted: dict[str, dict[str, int]] = defaultdict(dict)
        positional: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        title_inverted: dict[str, dict[str, int]] = defaultdict(dict)
        title_doc_len: dict[str, int] = {}
        doc_len: dict[str, int] = {}
        doc_store: dict[str, dict[str, str]] = {}

        for doc in documents:
            full_text = " ".join(part for part in [doc.title, doc.text] if part)
            tokens = tokenize(full_text)
            title_tokens = tokenize(doc.title)
            doc_len[doc.doc_id] = len(tokens)
            title_doc_len[doc.doc_id] = len(title_tokens)
            doc_store[doc.doc_id] = asdict(doc)

            counts = Counter(tokens)
            for term, tf in counts.items():
                inverted[term][doc.doc_id] = int(tf)
            for pos, term in enumerate(tokens):
                positional[term][doc.doc_id].append(pos)
            title_counts = Counter(title_tokens)
            for term, tf in title_counts.items():
                title_inverted[term][doc.doc_id] = int(tf)

        n_docs = len(documents)
        avgdl = sum(doc_len.values()) / n_docs if n_docs else 0.0
        title_avgdl = sum(title_doc_len.values()) / n_docs if n_docs else 0.0
        frozen_inverted = {term: dict(postings) for term, postings in inverted.items()}
        frozen_positional = {
            term: {doc_id: positions for doc_id, positions in postings.items()}
            for term, postings in positional.items()
        }
        frozen_title_inverted = {term: dict(postings) for term, postings in title_inverted.items()}
        return cls(
            inverted_index=frozen_inverted,
            positional_index=frozen_positional,
            title_inverted_index=frozen_title_inverted,
            title_df={term: len(postings) for term, postings in frozen_title_inverted.items()},
            title_doc_len=title_doc_len,
            title_avgdl=title_avgdl,
            df={term: len(postings) for term, postings in frozen_inverted.items()},
            doc_len=doc_len,
            doc_store=doc_store,
            avgdl=avgdl,
            n_docs=n_docs,
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


def _dump(obj: object, path: Path) -> None:
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


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
