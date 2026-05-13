from __future__ import annotations

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = CURRENT_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from bm25.bm25_engine import BM25SearchEngine
from io import read_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="search_tfidf.build_bm25_index",
        description="Build and save a BM25 inverted index.",
    )
    parser.add_argument("--input", required=True, help="Input JSONL corpus path")
    parser.add_argument("--model-dir", required=True, help="Directory to store BM25 artifacts")
    parser.add_argument("--champion-size", type=int, default=8000, help="Optional champion list size per term")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    docs = read_jsonl(args.input)
    engine = BM25SearchEngine(champion_size=args.champion_size)
    engine.fit(docs)
    engine.save(args.model_dir)
    print(f"Built BM25 index for {len(docs)} documents at {args.model_dir}")


if __name__ == "__main__":
    main()
