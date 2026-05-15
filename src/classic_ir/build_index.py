from __future__ import annotations

from pathlib import Path
import sys


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classic_ir.index import ClassicIndex


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
BM25_MODEL_DIR = ROOT / "artifacts" / "bm25_legal_full"
TOKENIZED_CORPUS_PATH = ROOT / "artifacts" / "bm25_underthesea" / "corpus_doc_id.jsonl"
INDEX_DIR = ROOT / "artifacts" / "classic_ir"


def main() -> None:
    index = ClassicIndex.from_bm25_model(
        BM25_MODEL_DIR,
        tokenized_corpus_path=TOKENIZED_CORPUS_PATH,
        positional_index_path=INDEX_DIR / "positional_index.pkl",
    )
    index.save(INDEX_DIR)

    print(f"BM25 source: {BM25_MODEL_DIR}")
    print(f"Token positions: {TOKENIZED_CORPUS_PATH}")
    print(f"Saved classic IR index to: {INDEX_DIR}")
    print(f"N={index.n_docs} avgdl={index.avgdl:.2f} vocab={len(index.inverted_index)}")
    print(f"positional_terms={len(index.positional_index)}")


if __name__ == "__main__":
    main()
