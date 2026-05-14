from __future__ import annotations

import re
import unicodedata


_PUNCT_RE = re.compile(r"[^\w\s/.-]+", re.UNICODE)
_SPACES_RE = re.compile(r"\s+")

LEGAL_KEEP_TERMS = {
    "bảo_hiểm",
    "đất_đai",
    "điều",
    "hợp_đồng",
    "khoản",
    "luật",
    "nghị_định",
    "nghĩa_vụ",
    "quyền",
    "sử_dụng",
    "thông_tư",
    "trách_nhiệm",
}

STOPWORDS = {
    "bị",
    "bởi",
    "các",
    "cái",
    "cần",
    "có",
    "của",
    "cho",
    "đã",
    "đang",
    "đây",
    "để",
    "đến",
    "được",
    "hay",
    "khi",
    "là",
    "lại",
    "mà",
    "một",
    "này",
    "nên",
    "nếu",
    "những",
    "như",
    "ở",
    "ra",
    "sẽ",
    "sự",
    "tại",
    "thì",
    "theo",
    "trong",
    "trên",
    "từ",
    "và",
    "về",
    "việc",
    "với",
}


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def normalize_text(text: str) -> str:
    text = normalize_unicode(text)
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = text.replace("_", " ")
    text = _SPACES_RE.sub(" ", text).strip()
    return text


def word_tokenize_vi(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []

    try:
        from underthesea import word_tokenize

        tokenized = word_tokenize(text, format="text")
        return [token for token in tokenized.split() if token]
    except Exception:
        return text.split()


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    tokens = word_tokenize_vi(text)
    if not remove_stopwords:
        return tokens

    filtered: list[str] = []
    for token in tokens:
        if token in LEGAL_KEEP_TERMS:
            filtered.append(token)
            continue
        if token in STOPWORDS:
            continue
        filtered.append(token)
    return filtered


def query_terms(text: str) -> list[str]:
    return tokenize(text, remove_stopwords=True)
