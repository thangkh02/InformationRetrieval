from __future__ import annotations

import sys
from pathlib import Path


CHAMPION_ROOT = Path(__file__).resolve().parents[1] / "champion-list"
if str(CHAMPION_ROOT) not in sys.path:
    sys.path.insert(0, str(CHAMPION_ROOT))

from text_utils import normalize_text, tokenize_underthesea_text


def tokenize(text: str) -> list[str]:
    return list(tokenize_underthesea_text(text))


def query_terms(text: str) -> list[str]:
    return tokenize(text)
