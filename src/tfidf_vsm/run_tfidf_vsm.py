"""
TF-IDF + Vector Space Model — pure Python, no external ML libraries.

Implements:
  - Text normalisation (lowercase, unicode NFKD, strip accents & punctuation)
  - N-gram extraction
  - TF-IDF weighting  : tf = 1 + log(raw_tf)   idf = log((N+1)/(df+1)) + 1
  - L2-normalised document vectors stored as sparse dicts
  - Inverted index for fast cosine-similarity retrieval
  - Serialisation via stdlib `pickle`

CLI subcommands
---------------
  build   --corpus <jsonl>  --output-dir <dir>  [--max-features N]
                            [--min-df N]         [--ngram-max N]
  search  --index <dir>     --query <text>       [--top-k N]
  eval    --index <dir>     --queries <jsonl>    --qrels <jsonl>
                            [--k N]              [--log-every N]
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Text processing helpers (pure Python)
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """Lowercase → NFKD unicode → strip combining chars → strip punctuation."""
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_terms(text: str, ngram_range: tuple[int, int] = (1, 2)) -> list[str]:
    """Tokenise *text* and produce all n-grams in [ngram_range[0], ngram_range[1]]."""
    norm = _normalize_text(text)
    if not norm:
        return []
    tokens = norm.split()
    terms: list[str] = []
    n_min, n_max = ngram_range
    for n in range(n_min, n_max + 1):
        if n <= 0 or len(tokens) < n:
            continue
        if n == 1:
            terms.extend(tokens)
        else:
            for i in range(len(tokens) - n + 1):
                terms.append(" ".join(tokens[i : i + n]))
    return terms


def _l2_norm(vec: dict[int, float]) -> float:
    """Euclidean (L2) norm of a sparse vector represented as {index: value}."""
    return math.sqrt(sum(v * v for v in vec.values()))


# ---------------------------------------------------------------------------
# Index build
# ---------------------------------------------------------------------------

def build_index(
    corpus_path: str,
    output_dir: str,
    max_features: int = 200_000,
    min_df: int = 1,
    ngram_max: int = 2,
) -> None:
    """Read corpus JSONL, compute TF-IDF vectors, persist index to *output_dir*."""
    ngram_range = (1, ngram_max)

    # --- load corpus ---------------------------------------------------------
    print(f"[build] Reading corpus: {corpus_path}")
    doc_ids: list[str] = []
    raw_texts: list[str] = []
    with Path(corpus_path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            doc_id = str(row.get("_id", row.get("doc_id", ""))).strip()
            title = str(row.get("title", ""))
            text = str(row.get("text", row.get("content", "")))
            merged = (title + " " + text).strip()
            if not doc_id or not merged:
                continue
            doc_ids.append(doc_id)
            raw_texts.append(merged)

    if not raw_texts:
        raise ValueError("No valid documents found in corpus.")
    print(f"[build] {len(doc_ids)} documents loaded.")

    # --- term frequency per document -----------------------------------------
    print("[build] Extracting terms …")
    doc_term_counts: list[Counter[str]] = []
    df_counter: Counter[str] = Counter()

    for text in raw_texts:
        counts: Counter[str] = Counter(_extract_terms(text, ngram_range))
        doc_term_counts.append(counts)
        df_counter.update(counts.keys())

    # --- build vocabulary (sorted by df desc, then alpha) --------------------
    filtered = [t for t, df in df_counter.items() if df >= min_df]
    filtered.sort(key=lambda t: (-df_counter[t], t))
    if max_features > 0:
        filtered = filtered[:max_features]
    vocab: dict[str, int] = {term: idx for idx, term in enumerate(filtered)}
    print(f"[build] Vocabulary size: {len(vocab)}")

    # --- IDF -----------------------------------------------------------------
    N = len(doc_ids)
    idf: list[float] = [0.0] * len(vocab)
    for term, idx in vocab.items():
        df = df_counter[term]
        idf[idx] = math.log((N + 1.0) / (df + 1.0)) + 1.0

    # --- document TF-IDF vectors + inverted index ----------------------------
    print("[build] Computing TF-IDF vectors …")
    doc_vectors: list[dict[int, float]] = []
    doc_norms: list[float] = []
    postings: dict[int, list[tuple[int, float]]] = defaultdict(list)

    for doc_idx, counts in enumerate(doc_term_counts):
        vec: dict[int, float] = {}
        for term, tf in counts.items():
            term_idx = vocab.get(term)
            if term_idx is None:
                continue
            # sublinear TF scaling: 1 + log(tf)
            vec[term_idx] = (1.0 + math.log(tf)) * idf[term_idx]

        norm = _l2_norm(vec)
        doc_vectors.append(vec)
        doc_norms.append(norm)

        if norm > 0.0:
            for term_idx, weight in vec.items():
                postings[term_idx].append((doc_idx, weight))

    inverted_index: dict[int, list[tuple[int, float]]] = dict(postings)

    # --- persist -------------------------------------------------------------
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / "tfidf_vsm.pkl"
    payload = {
        "ngram_range": ngram_range,
        "max_features": max_features,
        "min_df": min_df,
        "doc_ids": doc_ids,
        "doc_texts": raw_texts,
        "vocab": vocab,
        "idf": idf,
        "doc_vectors": doc_vectors,
        "doc_norms": doc_norms,
        "inverted_index": inverted_index,
    }
    with open(index_path, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[build] Index saved → {index_path}")


# ---------------------------------------------------------------------------
# Index load
# ---------------------------------------------------------------------------

def _load_index(index_dir: str) -> dict:
    """Deserialise the index pickle from *index_dir*."""
    path = Path(index_dir) / "tfidf_vsm.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Index not found: {path}")
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ---------------------------------------------------------------------------
# Search (cosine similarity via inverted index)
# ---------------------------------------------------------------------------

def _search_raw(
    payload: dict,
    query: str,
    top_k: int,
) -> list[tuple[str, float]]:
    """
    Returns a list of (doc_id, cosine_score) sorted by score descending.
    All arithmetic is pure Python — no numpy.
    """
    vocab: dict[str, int] = payload["vocab"]
    idf: list[float] = payload["idf"]
    doc_norms: list[float] = payload["doc_norms"]
    doc_ids: list[str] = payload["doc_ids"]
    inverted_index: dict[int, list[tuple[int, float]]] = payload["inverted_index"]
    ngram_range: tuple[int, int] = tuple(payload.get("ngram_range", (1, 2)))

    if not query.strip():
        return []

    # build query TF-IDF vector
    q_counts: Counter[str] = Counter(_extract_terms(query, ngram_range))
    if not q_counts:
        return []

    q_vec: dict[int, float] = {}
    for term, tf in q_counts.items():
        term_idx = vocab.get(term)
        if term_idx is None:
            continue
        q_vec[term_idx] = (1.0 + math.log(tf)) * idf[term_idx]

    if not q_vec:
        return []

    q_norm = _l2_norm(q_vec)
    if q_norm == 0.0:
        return []

    # accumulate dot products via inverted index
    dot_scores: dict[int, float] = defaultdict(float)
    for term_idx, q_weight in q_vec.items():
        for doc_idx, d_weight in inverted_index.get(term_idx, []):
            dot_scores[doc_idx] += q_weight * d_weight

    # cosine similarity = dot / (q_norm * d_norm)
    scored: list[tuple[int, float]] = []
    for doc_idx, dot in dot_scores.items():
        d_norm = doc_norms[doc_idx]
        if d_norm == 0.0:
            continue
        score = dot / (q_norm * d_norm)
        if score > 0.0:
            scored.append((doc_idx, score))

    if not scored:
        return []

    # partial sort: keep only top_k best (manual insertion-sort free approach)
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[: min(top_k, len(scored))]
    return [(doc_ids[i], float(s)) for i, s in top]


def search(index_dir: str, query: str, top_k: int) -> list[tuple[str, float]]:
    payload = _load_index(index_dir)
    return _search_raw(payload, query, top_k)


# ---------------------------------------------------------------------------
# Evaluation (Recall@k, Hit@k, MRR@k)
# ---------------------------------------------------------------------------

def _load_jsonl(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def evaluate(
    index_dir: str,
    queries_path: str,
    qrels_path: str,
    k: int = 10,
    log_every: int = 100,
) -> None:
    print(f"[eval] Loading index   : {index_dir}")
    payload = _load_index(index_dir)
    print(f"[eval] Loading queries : {queries_path}")
    raw_queries = _load_jsonl(queries_path)
    print(f"[eval] Loading qrels   : {qrels_path}")
    raw_qrels = _load_jsonl(qrels_path)

    # --- build relevance map -------------------------------------------------
    rel_map: dict[str, set[str]] = defaultdict(set)
    for row in raw_qrels:
        qid = str(row.get("query-id", "")).strip()
        cid = str(row.get("corpus-id", "")).strip()
        score = float(row.get("score", 0))
        if qid and cid and score > 0:
            rel_map[qid].add(cid)

    # --- deduplicate queries -------------------------------------------------
    query_map: dict[str, str] = {}
    for q in raw_queries:
        qid = str(q.get("_id", "")).strip()
        text = str(q.get("text", "")).strip()
        if not qid or not text:
            continue
        if qid not in query_map:
            query_map[qid] = text

    eval_ids = [qid for qid in rel_map if qid in query_map]
    total_candidates = len(eval_ids)
    print(f"[eval] Queries total   : {len(raw_queries)}")
    print(f"[eval] Queries unique  : {len(query_map)}")
    print(f"[eval] Qrels rows      : {len(raw_qrels)}")
    print(f"[eval] Qrels unique ids: {len(rel_map)}")
    print(f"[eval] Evaluable       : {total_candidates}")

    if total_candidates == 0:
        raise ValueError("No evaluable queries found. Check queries/qrels alignment.")

    # --- evaluation loop -----------------------------------------------------
    total = 0
    recall_sum = 0.0
    hit_count = 0
    mrr_sum = 0.0

    for qid in eval_ids:
        text = query_map[qid]
        total += 1
        preds = _search_raw(payload, text, k)
        pred_ids = [doc_id for doc_id, _ in preds]

        rel_docs = rel_map[qid]
        retrieved_relevant = 0
        first_rank = 0
        for rank, doc_id in enumerate(pred_ids, start=1):
            if doc_id in rel_docs:
                retrieved_relevant += 1
                if first_rank == 0:
                    first_rank = rank

        if rel_docs:
            recall_sum += retrieved_relevant / len(rel_docs)
        if retrieved_relevant > 0:
            hit_count += 1
        if first_rank > 0:
            mrr_sum += 1.0 / first_rank

        if log_every > 0 and (total % log_every == 0 or total == total_candidates):
            print(
                f"[eval] {total}/{total_candidates} | "
                f"Recall@{k}={recall_sum / total:.4f}  "
                f"Hit@{k}={hit_count / total:.4f}  "
                f"MRR@{k}={mrr_sum / total:.4f}"
            )

    recall_at_k = recall_sum / total
    hit_at_k = hit_count / total
    mrr_at_k = mrr_sum / total

    print("-" * 50)
    print(f"Queries evaluated : {total}")
    print(f"Recall@{k}        : {recall_at_k:.4f}")
    print(f"Hit@{k}           : {hit_at_k:.4f}")
    print(f"MRR@{k}           : {mrr_at_k:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TF-IDF Vector Space Model — pure Python, no ML libraries"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Build TF-IDF index from corpus JSONL")
    p_build.add_argument("--corpus", required=True, help="Path to corpus .jsonl")
    p_build.add_argument("--output-dir", default="artifacts/tfidf_vsm", help="Directory to write index")
    p_build.add_argument("--max-features", type=int, default=200_000, help="Max vocabulary size (0 = unlimited)")
    p_build.add_argument("--min-df", type=int, default=1, help="Min document frequency for a term")
    p_build.add_argument("--ngram-max", type=int, default=2, help="Max n-gram size (1 = unigrams only)")

    p_search = sub.add_parser("search", help="Search top-k documents for a query")
    p_search.add_argument("--index", required=True, help="Path to index directory")
    p_search.add_argument("--query", required=True, help="Query string")
    p_search.add_argument("--top-k", type=int, default=10, help="Number of results to return")

    p_eval = sub.add_parser("eval", help="Evaluate Recall@k / Hit@k / MRR@k")
    p_eval.add_argument("--index", required=True, help="Path to index directory")
    p_eval.add_argument("--queries", required=True, help="Path to queries .jsonl")
    p_eval.add_argument("--qrels", required=True, help="Path to qrels .jsonl")
    p_eval.add_argument("--k", type=int, default=10, help="Cut-off rank")
    p_eval.add_argument("--log-every", type=int, default=100, help="Log progress every N queries")

    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.command == "build":
        build_index(
            corpus_path=args.corpus,
            output_dir=args.output_dir,
            max_features=args.max_features,
            min_df=args.min_df,
            ngram_max=args.ngram_max,
        )

    elif args.command == "search":
        results = search(index_dir=args.index, query=args.query, top_k=args.top_k)
        if not results:
            print("No results.")
            return
        for rank, (doc_id, score) in enumerate(results, start=1):
            print(f"{rank:>3}. {doc_id}\t{score:.6f}")

    elif args.command == "eval":
        evaluate(
            index_dir=args.index,
            queries_path=args.queries,
            qrels_path=args.qrels,
            k=args.k,
            log_every=args.log_every,
        )


if __name__ == "__main__":
    main()
