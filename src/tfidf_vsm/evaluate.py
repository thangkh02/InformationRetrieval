from __future__ import annotations

from collections.abc import Callable

from .engine import TfidfVSMEngine
from .io import load_qrels_jsonl, load_queries_jsonl


def evaluate_recall_mrr(
    engine: TfidfVSMEngine,
    queries_path: str,
    qrels_path: str,
    k: int = 10,
    log_every: int = 100,
    progress_callback: Callable[[int, int, int, float], None] | None = None,
) -> tuple[int, float, float]:
    raw_queries = load_queries_jsonl(queries_path)
    qrels = load_qrels_jsonl(qrels_path)
    # Deduplicate queries by id and keep the first text occurrence.
    query_map: dict[str, str] = {}
    for q in raw_queries:
        if q.query_id not in query_map:
            query_map[q.query_id] = q.text

    # Evaluate strictly on query ids present in qrels.
    eval_ids = [qid for qid in qrels.keys() if qid in query_map]
    total_candidates = len(eval_ids)

    total = 0
    hits = 0
    mrr_sum = 0.0

    for qid in eval_ids:
        rel_docs = qrels[qid]
        q_text = query_map[qid]
        total += 1
        results = engine.search(q_text, top_k=k)

        first_rank = 0
        for rank, row in enumerate(results, start=1):
            if row.doc_id in rel_docs:
                first_rank = rank
                break

        if first_rank > 0:
            hits += 1
            mrr_sum += 1.0 / first_rank

        if log_every > 0 and (total % log_every == 0 or total == total_candidates):
            running_recall = hits / total
            if progress_callback is not None:
                progress_callback(total, total_candidates, hits, running_recall)

    if total == 0:
        raise ValueError("No query matched qrels.")

    return total, hits / total, mrr_sum / total

