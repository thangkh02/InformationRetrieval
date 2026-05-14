from __future__ import annotations

import argparse
import json
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
        description="Build and save a BM25 champion-list model from the legal corpus JSONL.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Raw legal corpus JSONL with _id/title/text")
    source.add_argument(
        "--corpus-tokenized",
        help="Tokenized legal corpus JSONL with doc_id/id/_id and tokens; skips tokenizing the corpus again",
    )
    parser.add_argument("--model-dir", required=True, help="Directory to store bm25_model.joblib")
    parser.add_argument("--champion-size", type=int, default=8000, help="Champion list size per term")
    return parser


def load_documents(input_path: str | Path) -> list[LegalDocument]:
    docs: list[LegalDocument] = []
    path = Path(input_path)

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            doc_id = item.get("_id", item.get("doc_id"))
            if doc_id is None:
                raise KeyError("Corpus records must contain '_id' or 'doc_id'.")

            docs.append(
                LegalDocument(
                    doc_id=str(doc_id),
                    title=str(item.get("title", "")),
                    content=str(item.get("text", item.get("content", ""))),
                )
            )

    if not docs:
        raise ValueError("Corpus is empty.")

    return docs


def main() -> None:
    args = build_parser().parse_args()
    doc_count = 0

    if args.input:
        docs = load_documents(args.input)
        doc_count = len(docs)
        engine = BM25SearchEngine(champion_size=args.champion_size)
        engine.fit(docs)
    else:
        doc_ids, doc_lengths, avgdl, inverted_index, idf = load_tokenized_corpus(args.corpus_tokenized)
        doc_count = len(doc_ids)
        engine = BM25SearchEngine(champion_size=args.champion_size)
        engine.documents = [LegalDocument(doc_id=doc_id, title="", content="") for doc_id in doc_ids]
        engine.doc_lengths = doc_lengths
        engine.avgdl = avgdl
        engine.inverted_index = inverted_index
        engine.idf = idf
        engine.champion_index = {}
        if args.champion_size > 0:
            engine.champion_index = build_champion_index(
                inverted_index=inverted_index,
                doc_lengths=doc_lengths,
                avgdl=avgdl,
                idf=idf,
                champion_size=args.champion_size,
                k1=engine.k1,
                b=engine.b,
            )

    engine.save(args.model_dir)
    print(f"Built BM25 model for {doc_count} documents at {args.model_dir}")


if __name__ == "__main__":
    main()
