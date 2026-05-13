from __future__ import annotations

from pathlib import Path
import json
from typing import Iterable

from .documents import NewsDocument


def write_jsonl(records: Iterable[dict], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(input_path: str | Path) -> list[NewsDocument]:
    path = Path(input_path)
    docs: list[NewsDocument] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            doc_id = item.get("doc_id", item.get("_id"))
            if doc_id is None:
                raise KeyError("Input JSONL must contain either 'doc_id' or '_id'.")
            text = str(item.get("content", item.get("text", "")))
            docs.append(
                NewsDocument(
                    doc_id=doc_id,
                    title=str(item.get("title", "")),
                    summary=str(item.get("summary", "")),
                    category=str(item.get("category", "")),
                    content=text,
                )
            )
    return docs
