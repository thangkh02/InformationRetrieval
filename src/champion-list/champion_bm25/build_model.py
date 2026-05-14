from __future__ import annotations

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = CURRENT_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from documents import LegalDocument
from champion_bm25.engine import BM25SearchEngine
from champion_bm25.indexing import build_champion_index, load_tokenized_corpus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_champion_bm25_model",
        description="Build and save a BM25 champion-list model from tokenized legal corpus JSONL.",
    )
    parser.add_argument("--corpus-tokenized", required=True, help="Tokenized legal corpus JSONL with doc_id and tokens")
    parser.add_argument("--model-dir", required=True, help="Directory to store bm25_model.joblib")
    parser.add_argument("--champion-size", type=int, default=8000, help="Champion list size per term")
    return parser


def build_from_tokenized_corpus(input_path: str | Path, champion_size: int) -> BM25SearchEngine:
    doc_ids, doc_lengths, avgdl, inverted_index, idf = load_tokenized_corpus(input_path)
    champion_index = build_champion_index(
        inverted_index=inverted_index,
        doc_lengths=doc_lengths,
        avgdl=avgdl,
        idf=idf,
        champion_size=champion_size,
    )

    engine = BM25SearchEngine(champion_size=champion_size)
    engine.documents = [
        LegalDocument(doc_id=doc_id, title="", content="")
        for doc_id in doc_ids
    ]
    engine.doc_lengths = doc_lengths
    engine.avgdl = avgdl
    engine.inverted_index = inverted_index
    engine.idf = idf
    engine.champion_index = champion_index
    return engine


def main() -> None:
    args = build_parser().parse_args()

    engine = build_from_tokenized_corpus(args.corpus_tokenized, args.champion_size)
    engine.save(args.model_dir)
    print(f"Built BM25 model for {len(engine.documents)} documents at {args.model_dir}")


if __name__ == "__main__":
    main()
