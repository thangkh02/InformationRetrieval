from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

CURRENT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = CURRENT_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from text_utils import tokenize_underthesea_text

TOP_KS = (1, 3, 5, 10, 20, 50, 100)


def read_jsonl(path: str | Path) -> Iterable[dict]:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_tokenized_corpus(path: str | Path):
    doc_ids: list[str] = []
    doc_lengths: list[int] = []
    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)

    for doc_idx, item in enumerate(read_jsonl(path)):
        doc_id = item.get("doc_id", item.get("_id"))
        if doc_id is None:
            raise KeyError("Tokenized corpus records must contain 'doc_id' or '_id'.")

        tokens = item.get("tokens")
        if not isinstance(tokens, list):
            raise TypeError("Tokenized corpus records must contain a list field named 'tokens'.")

        token_list = [str(tok) for tok in tokens if str(tok).strip()]
        doc_ids.append(str(doc_id))
        doc_lengths.append(len(token_list))

        term_freqs = Counter(token_list)
        for term, tf in term_freqs.items():
            postings[term].append((doc_idx, tf))

    if not doc_ids:
        raise ValueError("Corpus is empty.")

    avgdl = float(sum(doc_lengths) / len(doc_lengths))
    total_docs = len(doc_ids)
    inverted_index = dict(postings)
    idf = {
        term: math.log(1.0 + (total_docs - len(posting_list) + 0.5) / (len(posting_list) + 0.5))
        for term, posting_list in inverted_index.items()
    }
    return doc_ids, doc_lengths, avgdl, inverted_index, idf


def build_champion_index(
    inverted_index: dict[str, list[tuple[int, int]]],
    doc_lengths: list[int],
    avgdl: float,
    idf: dict[str, float],
    champion_size: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> dict[str, list[tuple[int, int]]]:
    champion_index: dict[str, list[tuple[int, int]]] = {}
    for term, posting_list in inverted_index.items():
        scored_postings: list[tuple[float, int, int]] = []
        idf_term = idf.get(term, 0.0)
        for doc_idx, tf in posting_list:
            dl = doc_lengths[doc_idx]
            denom_norm = k1 * (1.0 - b + b * (dl / avgdl)) if avgdl > 0 else k1
            numer = tf * (k1 + 1.0)
            denom = tf + denom_norm
            score = idf_term * (numer / denom)
            scored_postings.append((score, doc_idx, tf))

        scored_postings.sort(key=lambda item: item[0], reverse=True)
        champion_index[term] = [(doc_idx, tf) for _, doc_idx, tf in scored_postings[:champion_size]]

    return champion_index


def load_queries(path: str | Path, tokenized: bool):
    queries: dict[str, list[str]] = {}
    for item in read_jsonl(path):
        query_id = item.get("qid", item.get("query-id", item.get("_id")))
        if query_id is None:
            raise KeyError("Query records must contain 'qid', 'query-id', or '_id'.")

        if tokenized:
            tokens = item.get("tokens")
            if not isinstance(tokens, list):
                raise TypeError("Tokenized query records must contain a list field named 'tokens'.")
            query_terms = [str(tok) for tok in tokens if str(tok).strip()]
        else:
            text = str(item.get("text", ""))
            query_terms = list(tokenize_underthesea_text(text))

        queries[str(query_id)] = query_terms

    if not queries:
        raise ValueError("Query file is empty.")

    return queries


def load_qrels(path: str | Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = defaultdict(set)
    for item in read_jsonl(path):
        query_id = item.get("query-id", item.get("qid", item.get("_id")))
        corpus_id = item.get("corpus-id", item.get("doc_id"))
        if query_id is None or corpus_id is None:
            raise KeyError("Qrels records must contain query and corpus identifiers.")
        qrels[str(query_id)].add(str(corpus_id))
    if not qrels:
        raise ValueError("Qrels file is empty.")
    return qrels


def bm25_search(
    query_terms: list[str],
    doc_lengths: list[int],
    avgdl: float,
    inverted_index: dict[str, list[tuple[int, int]]],
    idf: dict[str, float],
    champion_index: dict[str, list[tuple[int, int]]] | None = None,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[int]:
    candidate_scores: dict[int, float] = defaultdict(float)
    posting_source = champion_index if champion_index is not None else inverted_index

    for term in set(query_terms):
        posting_list = posting_source.get(term)
        if not posting_list:
            continue

        idf_term = idf.get(term, 0.0)
        for doc_idx, tf in posting_list:
            dl = doc_lengths[doc_idx]
            denom_norm = k1 * (1.0 - b + b * (dl / avgdl)) if avgdl > 0 else k1
            numer = tf * (k1 + 1.0)
            denom = tf + denom_norm
            candidate_scores[doc_idx] += idf_term * (numer / denom)

    if not candidate_scores:
        return []

    ranked = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
    return [doc_idx for doc_idx, _ in ranked]


def compute_metrics(ranked_doc_ids: list[str], relevant_doc_ids: set[str]) -> dict[int, tuple[float, float, float, float]]:
    results: dict[int, tuple[float, float, float, float]] = {}
    relevant_count = len(relevant_doc_ids)

    for k in TOP_KS:
        top = ranked_doc_ids[:k]
        hits = [1 if doc_id in relevant_doc_ids else 0 for doc_id in top]

        recall = sum(hits) / relevant_count if relevant_count else 0.0
        hit = 1.0 if any(hits) else 0.0

        rr = 0.0
        for rank, is_hit in enumerate(hits, start=1):
            if is_hit:
                rr = 1.0 / rank
                break

        dcg = 0.0
        for rank, is_hit in enumerate(hits, start=1):
            if is_hit:
                dcg += 1.0 / math.log2(rank + 1)

        ideal_hits = min(relevant_count, k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1)) if ideal_hits else 0.0
        ndcg = dcg / idcg if idcg > 0 else 0.0

        results[k] = (recall, hit, rr, ndcg)

    return results


def aggregate_metrics(rows: list[dict[int, tuple[float, float, float, float]]]) -> dict[int, tuple[float, float, float, float]]:
    totals = {k: [0.0, 0.0, 0.0, 0.0] for k in TOP_KS}
    for row in rows:
        for k in TOP_KS:
            values = row[k]
            for idx in range(4):
                totals[k][idx] += values[idx]

    count = len(rows)
    return {k: tuple(value / count for value in totals[k]) for k in TOP_KS}


def evaluate(
    label: str,
    doc_ids: list[str],
    doc_lengths: list[int],
    avgdl: float,
    inverted_index: dict[str, list[tuple[int, int]]],
    idf: dict[str, float],
    queries: dict[str, list[str]],
    qrels: dict[str, set[str]],
    champion_index: dict[str, list[tuple[int, int]]] | None = None,
    top_k: int = 100,
) -> dict[str, object]:
    query_ids = [query_id for query_id in queries if query_id in qrels]
    if not query_ids:
        raise ValueError("No overlapping query ids between queries and qrels.")

    times: list[float] = []
    per_query_rows: list[dict[int, tuple[float, float, float, float]]] = []

    start_total = time.perf_counter()
    for query_id in query_ids:
        start = time.perf_counter()
        ranked_indices = bm25_search(
            queries[query_id],
            doc_lengths,
            avgdl,
            inverted_index,
            idf,
            champion_index=champion_index,
        )
        times.append(time.perf_counter() - start)
        ranked_doc_ids = [doc_ids[idx] for idx in ranked_indices[:top_k]]
        per_query_rows.append(compute_metrics(ranked_doc_ids, qrels[query_id]))
    total_time = time.perf_counter() - start_total

    metrics = aggregate_metrics(per_query_rows)
    return {
        "label": label,
        "query_count": len(query_ids),
        "latency_ms": 1000.0 * (total_time / len(query_ids)),
        "p50_ms": 1000.0 * statistics.median(times),
        "p95_ms": 1000.0 * sorted(times)[max(0, int(0.95 * len(times)) - 1)],
        "metrics": metrics,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate_bm25",
        description="Evaluate BM25 full scoring and champion list on tokenized or raw query files.",
    )
    parser.add_argument("--corpus-tokenized", required=True, help="Tokenized corpus JSONL with doc_id and tokens")
    parser.add_argument("--qrels", required=True, help="Qrels JSONL with query-id and corpus-id")
    parser.add_argument("--queries-tokenized", help="Tokenized query JSONL with qid and tokens")
    parser.add_argument("--queries-raw", help="Raw query JSONL with _id/qid and text")
    parser.add_argument("--champion-size", type=int, default=9000, help="Champion list size per term")
    parser.add_argument(
        "--mode",
        choices=["full", "champion", "both"],
        default="both",
        help="Run full BM25, champion list BM25, or both",
    )
    parser.add_argument("--top-k", type=int, default=100, help="Cutoff used for ranking and metric aggregation")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.queries_tokenized:
        queries = load_queries(args.queries_tokenized, tokenized=True)
    elif args.queries_raw:
        queries = load_queries(args.queries_raw, tokenized=False)
    else:
        raise ValueError("Provide either --queries-tokenized or --queries-raw.")

    qrels = load_qrels(args.qrels)
    doc_ids, doc_lengths, avgdl, inverted_index, idf = load_tokenized_corpus(args.corpus_tokenized)

    champion_index = None
    champion_build_seconds = 0.0
    if args.mode in {"champion", "both"}:
        start_build = time.perf_counter()
        champion_index = build_champion_index(
            inverted_index=inverted_index,
            doc_lengths=doc_lengths,
            avgdl=avgdl,
            idf=idf,
            champion_size=args.champion_size,
        )
        champion_build_seconds = time.perf_counter() - start_build

    print(f"corpus_docs={len(doc_ids)}")
    print(f"queries_loaded={len(queries)}")
    print(f"qrels_queries={len(qrels)}")
    if champion_index is not None:
        print(f"champion_size={args.champion_size}")
        print(f"champion_build_s={champion_build_seconds:.2f}")

    rows = []
    if args.mode in {"full", "both"}:
        rows.append(
            evaluate(
                label="BM25 full",
                doc_ids=doc_ids,
                doc_lengths=doc_lengths,
                avgdl=avgdl,
                inverted_index=inverted_index,
                idf=idf,
                queries=queries,
                qrels=qrels,
                champion_index=None,
                top_k=args.top_k,
            )
        )

    if args.mode in {"champion", "both"}:
        rows.append(
            evaluate(
                label=f"BM25 champion={args.champion_size}",
                doc_ids=doc_ids,
                doc_lengths=doc_lengths,
                avgdl=avgdl,
                inverted_index=inverted_index,
                idf=idf,
                queries=queries,
                qrels=qrels,
                champion_index=champion_index,
                top_k=args.top_k,
            )
        )

    print("model\tlatency_ms\tp50_ms\tp95_ms\tR@1\tR@10\tR@100\tMRR@10\tnDCG@10")
    for row in rows:
        metrics = row["metrics"]
        print(
            f"{row['label']}\t{row['latency_ms']:.2f}\t{row['p50_ms']:.2f}\t{row['p95_ms']:.2f}\t"
            f"{metrics[1][0]:.4f}\t{metrics[10][0]:.4f}\t{metrics[100][0]:.4f}\t{metrics[10][2]:.4f}\t{metrics[10][3]:.4f}"
        )


if __name__ == "__main__":
    main()
