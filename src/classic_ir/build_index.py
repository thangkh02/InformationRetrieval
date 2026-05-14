from __future__ import annotations

from pathlib import Path
import sys


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classic_ir.data_loader import read_corpus, read_qrels, read_queries
from classic_ir.index import ClassicIndex


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "zalo_ai_legal_text_retrieval_vn"
CORPUS_PATH = DATA_DIR / "corpus.jsonl"
QUERIES_PATH = DATA_DIR / "queries_no_question_mark.jsonl"
QRELS_PATH = DATA_DIR / "qrels" / "test.jsonl"
INDEX_DIR = ROOT / "artifacts" / "classic_ir"


def main() -> None:
    docs = read_corpus(CORPUS_PATH)
    doc_ids = {doc.doc_id for doc in docs}
    qrels = [qrel for qrel in read_qrels(QRELS_PATH) if qrel.relevance > 0 and qrel.doc_id in doc_ids]
    relevant_query_ids = {qrel.query_id for qrel in qrels}
    queries = [query for query in read_queries(QUERIES_PATH) if query.query_id in relevant_query_ids]

    print(f"Corpus: {len(docs)} documents")
    print(f"Evaluation queries: {len(queries)} queries")
    print(f"Evaluation qrels: {len(qrels)} relevant pairs")
    print(f"Corpus path: {CORPUS_PATH}")
    print(f"Queries path: {QUERIES_PATH}")
    print(f"Qrels path: {QRELS_PATH}")

    index = ClassicIndex.build(docs)
    index.save(INDEX_DIR)
    print(f"Saved classic IR index to: {INDEX_DIR}")
    print(f"N={index.n_docs} avgdl={index.avgdl:.2f} vocab={len(index.inverted_index)}")


if __name__ == "__main__":
    main()
