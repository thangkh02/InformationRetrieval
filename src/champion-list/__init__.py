from __future__ import annotations

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from documents import LegalDocument, SearchResult
from text_utils import normalize_text, prepare_phobert_text, tokenize_text, tokenize_underthesea_text
from champion_bm25.engine import BM25SearchEngine

__all__ = [
    "BM25SearchEngine",
    "LegalDocument",
    "SearchResult",
    "normalize_text",
    "prepare_phobert_text",
    "tokenize_text",
    "tokenize_underthesea_text",
]
