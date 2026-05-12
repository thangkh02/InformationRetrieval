from __future__ import annotations

import argparse
import json
from textwrap import shorten

from search_tfidf import TfidfSearchEngine, read_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ir-search",
        description="Simple TF-IDF retrieval system for binhvq-news-corpus.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build TF-IDF index")
    build_parser.add_argument("--input", required=True, help="Input JSONL path")
    build_parser.add_argument("--model-dir", required=True, help="Directory to store model artifacts")

    search_parser = subparsers.add_parser("search", help="Search with cosine similarity")
    search_parser.add_argument("--model-dir", required=True, help="Model directory")
    search_parser.add_argument("--query", required=True, help="Search query")
    search_parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")

    return parser


def _print_results(results) -> None:
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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "build":
        docs = read_jsonl(args.input)
        engine = TfidfSearchEngine(min_df=1 if len(docs) < 5 else 2)
        engine.fit(docs)
        engine.save(args.model_dir)
        print(f"Built index for {len(docs)} documents at {args.model_dir}")
        return

    if args.command == "search":
        engine = TfidfSearchEngine.load(args.model_dir)
        results = engine.search(args.query, top_k=args.top_k)
        _print_results(results)
        if results:
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
        return


if __name__ == "__main__":
    main()

