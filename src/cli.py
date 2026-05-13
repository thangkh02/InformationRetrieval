from __future__ import annotations

import argparse
import json
from textwrap import shorten

from search_tfidf import (
    BGEM3SearchEngine,
    BM25SearchEngine,
    PhoBERTSearchEngine,
    TfidfSearchEngine,
    read_jsonl,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ir-search",
        description="Simple TF-IDF retrieval system for binhvq-news-corpus.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build TF-IDF index")
    build_parser.add_argument("--input", required=True, help="Input JSONL path")
    build_parser.add_argument("--model-dir", required=True, help="Directory to store model artifacts")

    build_bm25_parser = subparsers.add_parser("build-bm25", help="Build BM25 index")
    build_bm25_parser.add_argument("--input", required=True, help="Input JSONL path")
    build_bm25_parser.add_argument("--model-dir", required=True, help="Directory to store model artifacts")

    build_phobert_parser = subparsers.add_parser("build-phobert", help="Build PhoBERT vector index")
    build_phobert_parser.add_argument("--input", required=True, help="Input JSONL path")
    build_phobert_parser.add_argument("--model-dir", required=True, help="Directory to store model artifacts")
    build_phobert_parser.add_argument("--model-name", default="vinai/phobert-base-v2", help="Hugging Face model name")
    build_phobert_parser.add_argument("--batch-size", type=int, default=16, help="Encoding batch size")
    build_phobert_parser.add_argument("--max-length", type=int, default=256, help="Maximum token length")

    build_bge_parser = subparsers.add_parser("build-bge", help="Build BGE-M3 vector index")
    build_bge_parser.add_argument("--input", required=True, help="Input JSONL path")
    build_bge_parser.add_argument("--model-dir", required=True, help="Directory to store model artifacts")
    build_bge_parser.add_argument("--model-name", default="BAAI/bge-m3", help="Hugging Face model name")
    build_bge_parser.add_argument("--batch-size", type=int, default=8, help="Encoding batch size")
    build_bge_parser.add_argument("--max-length", type=int, default=1024, help="Maximum token length")
    build_bge_parser.add_argument(
        "--index-type",
        default="flat",
        choices=["flat", "ivf_flat"],
        help="FAISS index type to build",
    )
    build_bge_parser.add_argument("--nlist", type=int, default=100, help="IVF cluster count")
    build_bge_parser.add_argument("--nprobe", type=int, default=10, help="IVF probe count")
    build_bge_parser.add_argument("--device", default=None, help="Device to use, e.g. cuda, cuda:0, or cpu")

    search_parser = subparsers.add_parser("search", help="Search with cosine similarity")
    search_parser.add_argument("--model-dir", required=True, help="Model directory")
    search_parser.add_argument("--query", required=True, help="Search query")
    search_parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")

    search_bm25_parser = subparsers.add_parser("search-bm25", help="Search with BM25")
    search_bm25_parser.add_argument("--model-dir", required=True, help="Model directory")
    search_bm25_parser.add_argument("--query", required=True, help="Search query")
    search_bm25_parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")

    search_phobert_parser = subparsers.add_parser("search-phobert", help="Search with PhoBERT embeddings")
    search_phobert_parser.add_argument("--model-dir", required=True, help="Model directory")
    search_phobert_parser.add_argument("--query", required=True, help="Search query")
    search_phobert_parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")

    search_bge_parser = subparsers.add_parser("search-bge", help="Search with BGE-M3 embeddings")
    search_bge_parser.add_argument("--model-dir", required=True, help="Model directory")
    search_bge_parser.add_argument("--query", required=True, help="Search query")
    search_bge_parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    search_bge_parser.add_argument("--device", default=None, help="Device to use, e.g. cuda, cuda:0, or cpu")

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

    if args.command == "build-bm25":
        docs = read_jsonl(args.input)
        engine = BM25SearchEngine()
        engine.fit(docs)
        engine.save(args.model_dir)
        print(f"Built BM25 index for {len(docs)} documents at {args.model_dir}")
        return

    if args.command == "build-phobert":
        docs = read_jsonl(args.input)
        engine = PhoBERTSearchEngine(
            model_name=args.model_name,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        engine.fit(docs)
        engine.save(args.model_dir)
        print(f"Built PhoBERT vector index for {len(docs)} documents at {args.model_dir}")
        return

    if args.command == "build-bge":
        docs = read_jsonl(args.input)
        engine = BGEM3SearchEngine(
            model_name=args.model_name,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
            index_type=args.index_type,
            nlist=args.nlist,
            nprobe=args.nprobe,
        )
        engine.fit(docs)
        engine.save(args.model_dir)
        print(f"Built BGE-M3 vector index for {len(docs)} documents at {args.model_dir}")
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

    if args.command == "search-bm25":
        engine = BM25SearchEngine.load(args.model_dir)
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

    if args.command == "search-phobert":
        engine = PhoBERTSearchEngine.load(args.model_dir)
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

    if args.command == "search-bge":
        engine = BGEM3SearchEngine.load(args.model_dir, device=args.device)
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
