from __future__ import annotations

import json
from pathlib import Path

from .documents import ZaloDocument, ZaloQuery


def load_corpus_jsonl(path: str | Path) -> list[ZaloDocument]:
    docs: list[ZaloDocument] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            doc_id = str(row.get("_id", "")).strip()
            title = str(row.get("title", ""))
            content = str(row.get("text", row.get("content", "")))
            if not doc_id:
                continue
            docs.append(ZaloDocument(doc_id=doc_id, title=title, content=content))
    return docs


def load_queries_jsonl(path: str | Path) -> list[ZaloQuery]:
    queries: list[ZaloQuery] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            query_id = str(row.get("_id", "")).strip()
            text = str(row.get("text", "")).strip()
            if not query_id or not text:
                continue
            queries.append(ZaloQuery(query_id=query_id, text=text))
    return queries


def load_qrels_jsonl(path: str | Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            query_id = str(row.get("query-id", "")).strip()
            corpus_id = str(row.get("corpus-id", "")).strip()
            score = float(row.get("score", 0))
            if not query_id or not corpus_id or score <= 0:
                continue
            if query_id not in qrels:
                qrels[query_id] = set()
            qrels[query_id].add(corpus_id)
    return qrels
