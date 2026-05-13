from __future__ import annotations

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from documents import NewsDocument, SearchResult
from io import read_jsonl, write_jsonl
from text_utils import normalize_text, prepare_phobert_text, tokenize_text
from bm25.bm25_engine import BM25SearchEngine
from embeddings.bge_m3_engine import BGEM3SearchEngine
from embeddings.phobert_engine import PhoBERTSearchEngine
from tfidf.engine import TfidfSearchEngine

__all__ = [
    "BGEM3SearchEngine",
    "BM25SearchEngine",
    "NewsDocument",
    "PhoBERTSearchEngine",
    "SearchResult",
    "TfidfSearchEngine",
    "normalize_text",
    "prepare_phobert_text",
    "read_jsonl",
    "tokenize_text",
    "write_jsonl",
]
