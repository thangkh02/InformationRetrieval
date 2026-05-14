from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import pickle
import sys


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
TOKENIZED_CORPUS = ROOT / "artifacts" / "bm25_underthesea" / "corpus_doc_id.jsonl"
INDEX_DIR = ROOT / "artifacts" / "classic_ir"


def main() -> None:
    positional_index: dict[str, dict[str, list[int]]] = defaultdict(dict)
    doc_count = 0

    with TOKENIZED_CORPUS.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            doc_id = str(item["doc_id"])
            tokens = [str(token) for token in item.get("tokens", []) if str(token).strip()]
            for pos, token in enumerate(tokens):
                positions = positional_index[token].setdefault(doc_id, [])
                positions.append(pos)
            doc_count += 1

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    output_path = INDEX_DIR / "positional_index.pkl"
    frozen = {term: dict(postings) for term, postings in positional_index.items()}
    with output_path.open("wb") as handle:
        pickle.dump(frozen, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved positional index to {output_path}")
    print(f"documents={doc_count}")
    print(f"terms={len(frozen)}")


if __name__ == "__main__":
    main()
