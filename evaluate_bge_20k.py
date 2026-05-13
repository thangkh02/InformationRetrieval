from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import faiss
import joblib
import numpy as np


def read_qrels(path: Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if float(item.get("score", 1)) > 0:
                qrels[item["query-id"]].add(item["corpus-id"])
    return dict(qrels)


def dcg(relevance: list[int]) -> float:
    return sum(rel / np.log2(rank + 2) for rank, rel in enumerate(relevance))


def evaluate(
    corpus_dir: Path,
    query_dir: Path,
    qrels_path: Path,
    top_ks: list[int],
) -> dict[str, float]:
    corpus_meta = joblib.load(corpus_dir / "bge_m3_meta.joblib")
    query_meta = joblib.load(query_dir / "bge_m3_meta.joblib")
    query_embeddings = np.load(query_dir / "bge_m3_embeddings.npy").astype(np.float32, copy=False)
    index = faiss.read_index(str(corpus_dir / "bge_m3.index"))

    corpus_ids = [str(doc["doc_id"]) for doc in corpus_meta["documents"]]
    query_ids = [str(doc["doc_id"]) for doc in query_meta["documents"]]
    qrels = read_qrels(qrels_path)

    eval_positions = [i for i, qid in enumerate(query_ids) if qid in qrels]
    if not eval_positions:
        raise ValueError(f"No query from {query_dir} has qrels in {qrels_path}")

    max_k = max(top_ks)
    scores, indices = index.search(query_embeddings[eval_positions], max_k)

    metrics: dict[str, float] = {}
    per_k = {
        k: {
            "recall": [],
            "hit": [],
            "mrr": [],
            "ndcg": [],
        }
        for k in top_ks
    }

    for row, query_pos in enumerate(eval_positions):
        qid = query_ids[query_pos]
        relevant = qrels[qid]
        retrieved = [corpus_ids[int(idx)] for idx in indices[row] if idx >= 0]

        for k in top_ks:
            top_docs = retrieved[:k]
            hits = [1 if doc_id in relevant else 0 for doc_id in top_docs]
            hit_count = sum(hits)

            recall = hit_count / len(relevant)
            hit = 1.0 if hit_count > 0 else 0.0
            rr = 0.0
            for rank, is_hit in enumerate(hits, start=1):
                if is_hit:
                    rr = 1.0 / rank
                    break

            ideal_hits = [1] * min(len(relevant), k)
            ideal_dcg = dcg(ideal_hits)
            ndcg = dcg(hits) / ideal_dcg if ideal_dcg > 0 else 0.0

            per_k[k]["recall"].append(recall)
            per_k[k]["hit"].append(hit)
            per_k[k]["mrr"].append(rr)
            per_k[k]["ndcg"].append(ndcg)

    metrics["queries_evaluated"] = float(len(eval_positions))
    metrics["qrels"] = float(sum(len(v) for v in qrels.values()))
    for k in top_ks:
        metrics[f"Recall@{k}"] = float(np.mean(per_k[k]["recall"]))
        metrics[f"Hit@{k}"] = float(np.mean(per_k[k]["hit"]))
        metrics[f"MRR@{k}"] = float(np.mean(per_k[k]["mrr"]))
        metrics[f"nDCG@{k}"] = float(np.mean(per_k[k]["ndcg"]))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BGE-M3 retrieval on the 20k legal subset.")
    parser.add_argument("--corpus-dir", default="artifacts/bge_m3_legal_20k")
    parser.add_argument("--query-dir", default="artifacts/bge_m3_queries_20k")
    parser.add_argument("--qrels", default="data/zalo_ai_legal_text_retrieval_vn/qrels/test_20k.jsonl")
    parser.add_argument("--top-k", default="1,3,5,10,20,50,100")
    args = parser.parse_args()

    top_ks = [int(k.strip()) for k in args.top_k.split(",") if k.strip()]
    metrics = evaluate(Path(args.corpus_dir), Path(args.query_dir), Path(args.qrels), top_ks)

    print(f"Qrels: {args.qrels}")
    print(f"Queries evaluated: {int(metrics.pop('queries_evaluated'))}")
    print(f"Relevant pairs: {int(metrics.pop('qrels'))}")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
