from __future__ import annotations

import argparse

from zalo_tfidf_vsm.engine import ZaloTfidfVSMEngine
from zalo_tfidf_vsm.evaluate import evaluate_recall_mrr
from zalo_tfidf_vsm.io import load_corpus_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Zalo TF-IDF + Vector Space Model")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build")
    p_build.add_argument("--corpus", required=True)
    p_build.add_argument("--model-dir", required=True)

    p_search = sub.add_parser("search")
    p_search.add_argument("--model-dir", required=True)
    p_search.add_argument("--query", required=True)
    p_search.add_argument("--top-k", type=int, default=5)

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--model-dir", required=True)
    p_eval.add_argument("--queries", required=True)
    p_eval.add_argument("--qrels", required=True)
    p_eval.add_argument("--k", type=int, default=10)
    p_eval.add_argument("--log-every", type=int, default=100)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "build":
        docs = load_corpus_jsonl(args.corpus)
        engine = ZaloTfidfVSMEngine(min_df=1)
        engine.fit(docs)
        engine.save(args.model_dir)
        print(f"Built index for {len(docs)} docs -> {args.model_dir}")
        return

    if args.command == "search":
        engine = ZaloTfidfVSMEngine.load(args.model_dir)
        rows = engine.search(args.query, top_k=args.top_k)
        for i, r in enumerate(rows, start=1):
            print(f"{i}. score={r.score:.6f} | id={r.doc_id}")
            print(f"   title: {r.title}")
        return

    if args.command == "eval":
        engine = ZaloTfidfVSMEngine.load(args.model_dir)

        def on_progress(done: int, total: int, hits: int, running_recall: float) -> None:
            print(f"[eval] {done}/{total} queries | hits={hits} | running_recall@{args.k}={running_recall:.4f}")

        total, recall_at_k, mrr_at_k = evaluate_recall_mrr(
            engine,
            args.queries,
            args.qrels,
            k=args.k,
            log_every=args.log_every,
            progress_callback=on_progress,
        )
        print(f"Queries evaluated: {total}")
        print(f"Recall@{args.k}: {recall_at_k:.4f}")
        print(f"MRR@{args.k}: {mrr_at_k:.4f}")


if __name__ == "__main__":
    main()
