from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import math
import sys


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classic_ir.data_loader import read_qrels, read_queries
from classic_ir.preprocess import query_terms
from classic_ir.search import ClassicSearchEngine, SearchResult


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "zalo_ai_legal_text_retrieval_vn"
QUERIES_PATH = DATA_DIR / "queries_no_question_mark.jsonl"
QRELS_PATH = DATA_DIR / "qrels" / "test.jsonl"
INDEX_DIR = ROOT / "artifacts" / "classic_ir"
BM25_MODEL_DIR = ROOT / "artifacts" / "bm25_legal_full"
TOKENIZED_CORPUS_PATH = ROOT / "artifacts" / "bm25_underthesea" / "corpus_doc_id.jsonl"


def group_qrels(path: Path) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for qrel in read_qrels(path):
        if qrel.relevance > 0:
            grouped[qrel.query_id].add(qrel.doc_id)
    return dict(grouped)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top = retrieved[:k]
    if not top:
        return 0.0
    return sum(1 for doc_id in top if doc_id in relevant) / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = retrieved[:k]
    return sum(1 for doc_id in top if doc_id in relevant) / len(relevant)


def reciprocal_rank_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def average_precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            hits += 1
            total += hits / rank
    return total / min(len(relevant), k)


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg = 0.0
    for idx, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(idx + 1)
    ideal_hits = min(len(relevant), k)
    ideal = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return dcg / ideal if ideal else 0.0


def overlap_or_rank(engine: ClassicSearchEngine, query: str, top_k: int) -> list[SearchResult]:
    terms = query_terms(query)
    doc_scores: dict[str, float] = defaultdict(float)
    for term in terms:
        for doc_id, tf in engine.index.inverted_index.get(term, {}).items():
            doc_scores[doc_id] += tf
    ranked = sorted(doc_scores.items(), key=lambda item: (-item[1], item[0]))
    return engine._format_results(ranked[:top_k], query)


def proximity_rank(engine: ClassicSearchEngine, query: str, top_k: int, distance: int = 5) -> list[SearchResult]:
    terms = query_terms(query)
    if len(terms) < 2:
        return []

    doc_scores: dict[str, float] = defaultdict(float)
    for left, right in zip(terms, terms[1:]):
        matches = engine.near_doc_ids(left, right, distance)
        for doc_id in matches:
            doc_scores[doc_id] += engine._bm25_term_score(left, doc_id)
            doc_scores[doc_id] += engine._bm25_term_score(right, doc_id)

    ranked = sorted(doc_scores.items(), key=lambda item: (-item[1], item[0]))
    return engine._format_results(ranked[:top_k], query)


def evaluate_method(
    name: str,
    engine: ClassicSearchEngine,
    queries,
    qrels: dict[str, set[str]],
    top_k: int,
) -> dict[str, float | str]:
    rows = []
    total = sum(1 for q in queries if qrels.get(q.query_id))
    print(f"[{name}] evaluating {total} queries...", flush=True)
    done = 0
    for query in queries:
        relevant = qrels.get(query.query_id)
        if not relevant:
            continue

        if name == "Boolean OR + overlap rank":
            results = overlap_or_rank(engine, query.text, top_k)
        elif name == "Phrase Search + BM25 rank":
            results = engine.phrase_search_ranked(query.text, top_k=top_k)
        elif name == "Proximity NEAR/5 + BM25 rank":
            results = proximity_rank(engine, query.text, top_k, distance=5)
        elif name == "BM25 + title boost":
            results = engine.fielded_bm25_search(query.text, top_k=top_k, title_weight=0.2, text_weight=1.0)
        else:
            results = engine.bm25_search(query.text, top_k=top_k)
        done += 1
        if done % 50 == 0 or done == total:
            print(f"  [{name}] {done}/{total}", flush=True)

        retrieved = [result.doc_id for result in results]
        rows.append(
            {
                "P@5": precision_at_k(retrieved, relevant, 5),
                "P@10": precision_at_k(retrieved, relevant, 10),
                "R@5": recall_at_k(retrieved, relevant, 5),
                "R@10": recall_at_k(retrieved, relevant, 10),
                "MRR@10": reciprocal_rank_at_k(retrieved, relevant, 10),
                "MAP@10": average_precision_at_k(retrieved, relevant, 10),
                "NDCG@10": ndcg_at_k(retrieved, relevant, 10),
            }
        )

    summary: dict[str, float | str] = {"Method": name, "Queries": float(len(rows))}
    for metric in ["P@5", "P@10", "R@5", "R@10", "MRR@10", "MAP@10", "NDCG@10"]:
        summary[metric] = sum(float(row[metric]) for row in rows) / len(rows) if rows else 0.0
    return summary


def print_table(rows: list[dict[str, float | str]]) -> None:
    headers = ["Method", "Queries", "P@10", "R@10", "MRR@10", "MAP@10", "NDCG@10"]
    print("\t".join(headers))
    for row in rows:
        values = []
        for header in headers:
            value = row[header]
            values.append(str(int(value)) if header == "Queries" else f"{value:.4f}" if isinstance(value, float) else str(value))
        print("\t".join(values))


def main() -> None:
    engine = ClassicSearchEngine.load_from_bm25_model(
        str(BM25_MODEL_DIR),
        tokenized_corpus_path=str(TOKENIZED_CORPUS_PATH),
        positional_index_path=str(INDEX_DIR / "positional_index.pkl"),
    )
    qrels = group_qrels(QRELS_PATH)
    queries = [query for query in read_queries(QUERIES_PATH) if query.query_id in qrels]
    methods = [
        "BM25",
        "BM25 + title boost",
        "Phrase Search + BM25 rank",
        "Proximity NEAR/5 + BM25 rank",
    ]
    rows = [evaluate_method(method, engine, queries, qrels, top_k=10) for method in methods]
    print_table(rows)


if __name__ == "__main__":
    main()
