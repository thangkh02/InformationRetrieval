from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(slots=True)
class Document:
    doc_id: str
    title: str
    text: str


@dataclass(slots=True)
class Query:
    query_id: str
    text: str


@dataclass(slots=True)
class Qrel:
    query_id: str
    doc_id: str
    relevance: int


def read_corpus(path: str | Path) -> list[Document]:
    docs: list[Document] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            doc_id = str(item.get("doc_id", item.get("_id", "")))
            if not doc_id:
                raise KeyError("Corpus record must contain 'doc_id' or '_id'.")
            docs.append(
                Document(
                    doc_id=doc_id,
                    title=str(item.get("title", "")),
                    text=str(item.get("text", item.get("content", ""))),
                )
            )
    return docs


def read_queries(path: str | Path) -> list[Query]:
    queries: list[Query] = []
    seen: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            query_id = str(item.get("query_id", item.get("_id", "")))
            if not query_id or query_id in seen:
                continue
            seen.add(query_id)
            queries.append(Query(query_id=query_id, text=str(item.get("text", item.get("query_text", "")))))
    return queries


def read_qrels(path: str | Path) -> list[Qrel]:
    qrels: list[Qrel] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            relevance = int(float(item.get("relevance", item.get("score", 1))))
            qrels.append(
                Qrel(
                    query_id=str(item.get("query_id", item.get("query-id", ""))),
                    doc_id=str(item.get("doc_id", item.get("corpus-id", ""))),
                    relevance=relevance,
                )
            )
    return qrels

