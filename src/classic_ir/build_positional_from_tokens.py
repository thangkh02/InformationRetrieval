from __future__ import annotations

from pathlib import Path
import pickle
import sys


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classic_ir.index import build_positional_index_from_tokens


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
TOKENIZED_CORPUS = ROOT / "artifacts" / "bm25_underthesea" / "corpus_doc_id.jsonl"
INDEX_DIR = ROOT / "artifacts" / "classic_ir"


def main() -> None:
    positional_index = build_positional_index_from_tokens(TOKENIZED_CORPUS)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    output_path = INDEX_DIR / "positional_index.pkl"
    with output_path.open("wb") as handle:
        pickle.dump(positional_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved positional index to {output_path}")
    print(f"terms={len(positional_index)}")


if __name__ == "__main__":
    main()
