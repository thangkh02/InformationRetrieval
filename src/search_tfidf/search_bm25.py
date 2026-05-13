from __future__ import annotations

import argparse
import json
from textwrap import shorten

from .bm25_engine import BM25SearchEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="search_tfidf.search_bm25",
        description="Search a prebuilt BM25 index.",
    )
    parser.add_argument("--model-dir", required=True, help="Directory containing bm25_model.joblib")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    engine = BM25SearchEngine.load(args.model_dir)
    results = engine.search(args.query, top_k=args.top_k)

    if not results:
        print("No results found.")
        return

    for rank, result in enumerate(results, start=1):
        print(f"{rank}. score={result.score:.4f} | id={result.doc_id} | category={result.category}")
        print(f"   title   : {result.title}")
        if result.summary:
            print(f"   summary : {shorten(result.summary, width=220, placeholder='...')}")
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
                "summary": results[0].summary,
                "category": results[0].category,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
