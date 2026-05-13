from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_jsonl(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_index(corpus_path: str, output_dir: str, max_features: int, min_df: int, ngram_max: int) -> None:
    corpus = load_jsonl(corpus_path)
    doc_ids: list[str] = []
    texts: list[str] = []

    for item in corpus:
        doc_id = str(item.get("_id", item.get("doc_id", "")))
        text = str(item.get("text", item.get("content", "")))
        title = str(item.get("title", ""))
        merged = (title + " " + text).strip()
        if not doc_id or not merged:
            continue
        doc_ids.append(doc_id)
        texts.append(merged)

    if not texts:
        raise ValueError("No valid documents found in corpus.")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, ngram_max),
        max_features=max_features,
        min_df=min_df,
    )
    doc_matrix = vectorizer.fit_transform(texts)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "doc_ids": doc_ids,
            "doc_texts": texts,
            "doc_matrix": doc_matrix,
        },
        out / "tfidf_vsm.joblib",
    )
    print(f"Built TF-IDF VSM index: {len(doc_ids)} docs -> {out / 'tfidf_vsm.joblib'}")


def load_index(index_path: str):
    payload = joblib.load(index_path)
    return payload["vectorizer"], payload["doc_ids"], payload["doc_texts"], payload["doc_matrix"]


def search(index_path: str, query: str, top_k: int) -> list[tuple[str, float]]:
    vectorizer, doc_ids, _, doc_matrix = load_index(index_path)
    q_vec = vectorizer.transform([query])
    if q_vec.nnz == 0:
        return []

    scores = cosine_similarity(q_vec, doc_matrix).ravel()
    if scores.size == 0:
        return []

    top_k = min(top_k, len(doc_ids))
    top_idx = np.argpartition(-scores, top_k - 1)[:top_k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return [(doc_ids[i], float(scores[i])) for i in top_idx]


def evaluate(index_path: str, queries_path: str, qrels_path: str, k: int) -> None:
    queries = load_jsonl(queries_path)
    qrels = load_jsonl(qrels_path)

    rel_map: dict[str, set[str]] = defaultdict(set)
    for row in qrels:
        qid = str(row.get("query-id", ""))
        cid = str(row.get("corpus-id", ""))
        score = float(row.get("score", 0))
        if qid and cid and score > 0:
            rel_map[qid].add(cid)

    total = 0
    hit_count = 0
    mrr_sum = 0.0

    for q in queries:
        qid = str(q.get("_id", ""))
        text = str(q.get("text", "")).strip()
        if not qid or not text or qid not in rel_map:
            continue

        total += 1
        preds = search(index_path=index_path, query=text, top_k=k)
        pred_ids = [doc_id for doc_id, _ in preds]

        rel_docs = rel_map[qid]
        found_rank = 0
        for rank, doc_id in enumerate(pred_ids, start=1):
            if doc_id in rel_docs:
                found_rank = rank
                break

        if found_rank > 0:
            hit_count += 1
            mrr_sum += 1.0 / found_rank

    if total == 0:
        raise ValueError("No evaluable queries found. Check queries/qrels alignment.")

    recall_at_k = hit_count / total
    mrr_at_k = mrr_sum / total

    print(f"Queries evaluated : {total}")
    print(f"Recall@{k}         : {recall_at_k:.4f}")
    print(f"MRR@{k}            : {mrr_at_k:.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TF-IDF + Vector Space Model ")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Build index from corpus")
    p_build.add_argument("--corpus", required=True)
    p_build.add_argument("--output-dir", default="artifacts/tfidf_vsm")
    p_build.add_argument("--max-features", type=int, default=200000)
    p_build.add_argument("--min-df", type=int, default=1)
    p_build.add_argument("--ngram-max", type=int, default=2)

    p_search = sub.add_parser("search", help="Search top-k documents")
    p_search.add_argument("--index", required=True)
    p_search.add_argument("--query", required=True)
    p_search.add_argument("--top-k", type=int, default=10)

    p_eval = sub.add_parser("eval", help="Evaluate with qrels")
    p_eval.add_argument("--index", required=True)
    p_eval.add_argument("--queries", required=True)
    p_eval.add_argument("--qrels", required=True)
    p_eval.add_argument("--k", type=int, default=10)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "build":
        build_index(
            corpus_path=args.corpus,
            output_dir=args.output_dir,
            max_features=args.max_features,
            min_df=args.min_df,
            ngram_max=args.ngram_max,
        )
        return

    if args.command == "search":
        rows = search(index_path=args.index, query=args.query, top_k=args.top_k)
        if not rows:
            print("No results")
            return
        for i, (doc_id, score) in enumerate(rows, start=1):
            print(f"{i}. {doc_id}\t{score:.6f}")
        return

    if args.command == "eval":
        evaluate(index_path=args.index, queries_path=args.queries, qrels_path=args.qrels, k=args.k)


if __name__ == "__main__":
    main()

