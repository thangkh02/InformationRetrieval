from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from textwrap import shorten

CURRENT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = CURRENT_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from champion_bm25.engine import BM25SearchEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="champion_bm25.search",
        description="Search with a prebuilt BM25 champion-list model.",
    )
    parser.add_argument("--model-dir", required=True, help="Directory containing bm25_model.joblib")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    return parser


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = build_parser().parse_args()

    try:
        engine = BM25SearchEngine.load(args.model_dir)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{exc}. Build the model first with: "
            f"python src/champion-list/champion_bm25/build_model.py "
            f"--corpus-tokenized artifacts/bm25_underthesea/corpus_doc_id.jsonl "
            f"--model-dir {args.model_dir}"
        ) from exc
    results = engine.search(args.query, top_k=args.top_k)

    if not results:
        print("No results found.")
        return

    for rank, result in enumerate(results, start=1):
        print(f"{rank}. score={result.score:.4f} | id={result.doc_id}")
        if result.title:
            print(f"   title   : {result.title}")
        if result.content:
            print(f"   content : {shorten(result.content, width=220, placeholder='...')}")
        print()

    print("Top result JSON:")
    print(
        json.dumps(
            {
                "doc_id": results[0].doc_id,
                "score": results[0].score,
                "title": results[0].title,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
