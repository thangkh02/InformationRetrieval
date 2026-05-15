from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent
SRC_ROOT = ROOT / "src"
CHAMPION_SRC = SRC_ROOT / "champion-list"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(CHAMPION_SRC) not in sys.path:
    sys.path.insert(0, str(CHAMPION_SRC))

from champion_bm25.engine import BM25SearchEngine  # noqa: E402
from classic_ir.search import ClassicSearchEngine  # noqa: E402
from tfidf_vsm.documents import Document as TfidfDocument  # noqa: E402
from tfidf_vsm.engine import TfidfVSMEngine  # noqa: E402


BM25_DEFAULT_MODEL_DIR = ROOT / "artifacts" / "bm25_legal_6000"
TFIDF_DEFAULT_MODEL_DIR = ROOT / "artifacts" / "tfidf_vsm_legal"
CLASSIC_BM25_MODEL_DIR = ROOT / "artifacts" / "bm25_legal_6000"
CLASSIC_INDEX_DIR = ROOT / "artifacts" / "classic_ir"
TOKENIZED_CORPUS_PATH = ROOT / "artifacts" / "bm25_underthesea" / "corpus_doc_id.jsonl"
CORPUS_PATH = ROOT / "data" / "zalo_ai_legal_text_retrieval_vn" / "corpus.jsonl"


st.set_page_config(page_title="Legal Search", page_icon="IR", layout="wide")

st.title("Legal Search")
st.caption("Tim kiem tren legal corpus bang BM25 full, BM25 champion-list hoac TF-IDF VSM.")


@st.cache_resource
def load_bm25_engine(model_dir: str) -> BM25SearchEngine:
    return BM25SearchEngine.load(model_dir)


@st.cache_resource
def load_tfidf_engine(model_dir: str) -> TfidfVSMEngine:
    model_path = Path(model_dir) / "tfidf_vsm.pkl"
    if model_path.exists():
        return TfidfVSMEngine.load(model_dir)

    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"Khong tim thay corpus tai {CORPUS_PATH}")

    documents: list[TfidfDocument] = []
    with CORPUS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            doc_id = item.get("_id", item.get("doc_id"))
            if doc_id is None:
                continue
            documents.append(
                TfidfDocument(
                    doc_id=str(doc_id),
                    title=str(item.get("title", "")),
                    content=str(item.get("text", item.get("content", ""))),
                )
            )

    if not documents:
        raise ValueError("Corpus is empty, cannot build TF-IDF model.")

    engine = TfidfVSMEngine(min_df=2 if len(documents) >= 5 else 1)
    engine.fit(documents)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    engine.save(model_dir)
    return engine


@st.cache_resource
def load_classic_engine(
    bm25_model_dir: str,
    tokenized_corpus_path: str,
    positional_index_path: str,
) -> ClassicSearchEngine:
    return ClassicSearchEngine.load_from_bm25_model(
        bm25_model_dir,
        tokenized_corpus_path=tokenized_corpus_path,
        positional_index_path=positional_index_path,
    )


@st.cache_data
def load_doc_metadata() -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    if not CORPUS_PATH.exists():
        return metadata

    with CORPUS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            doc_id = item.get("_id", item.get("doc_id"))
            if doc_id is None:
                continue
            metadata[str(doc_id)] = {
                "title": str(item.get("title", "")),
                "content": str(item.get("text", item.get("content", ""))),
            }
    return metadata


def search_bm25(engine: BM25SearchEngine, query: str, top_k: int, use_champion: bool) -> list:
    original_champion_size = engine.champion_size
    try:
        if not use_champion:
            engine.champion_size = 0
        return engine.search(query, top_k=top_k)
    finally:
        engine.champion_size = original_champion_size


def render_results(results: list) -> None:
    if not results:
        st.info("Khong tim thay ket qua phu hop.")
        return

    doc_metadata = load_doc_metadata()
    st.subheader(f"Ket qua top-{len(results)}")
    for rank, result in enumerate(results, start=1):
        meta = doc_metadata.get(str(result.doc_id), {})
        title = meta.get("title") or result.title or str(result.doc_id)
        content = meta.get("content") or result.content or ""
        with st.container(border=True):
            st.markdown(f"**{rank}. {title}**")
            st.write(f"Score: `{result.score:.4f}` | Doc ID: `{result.doc_id}`")
            if content:
                st.write(content[:500] + ("..." if len(content) > 500 else ""))


def render_classic_results(results: list) -> None:
    if not results:
        st.info("Khong tim thay ket qua phu hop.")
        return

    doc_metadata = load_doc_metadata()
    st.subheader(f"Ket qua top-{len(results)}")
    for result in results:
        meta = doc_metadata.get(str(result.doc_id), {})
        title = meta.get("title") or result.title or str(result.doc_id)
        content = meta.get("content") or ""
        with st.container(border=True):
            st.markdown(f"**{result.rank}. {title}**")
            st.write(f"Score: `{result.score:.4f}` | Doc ID: `{result.doc_id}`")
            if getattr(result, "snippet", ""):
                st.write(result.snippet)
            elif content:
                st.write(content[:500] + ("..." if len(content) > 500 else ""))


with st.sidebar:
    st.header("Thiet lap")
    search_mode = st.radio(
        "Che do search",
        ["BM25 champion", "BM25 full", "TF-IDF VSM", "Classic IR Proximity"],
        index=0,
    )
    top_k = st.slider("Top-k", min_value=1, max_value=20, value=5, step=1)

    if search_mode.startswith("BM25") or search_mode == "Classic IR Proximity":
        model_dir = st.text_input("BM25 model dir", value=str(BM25_DEFAULT_MODEL_DIR), key="bm25_model_dir")
    elif search_mode == "TF-IDF VSM":
        model_dir = st.text_input("TF-IDF model dir", value=str(TFIDF_DEFAULT_MODEL_DIR), key="tfidf_model_dir")
        st.caption("Neu chua co model, app se build TF-IDF mot lan tu corpus legal.")
    else:
        model_dir = st.text_input("Classic model dir", value=str(BM25_DEFAULT_MODEL_DIR), key="classic_model_dir")

    st.write(f"Model: `{Path(model_dir).name}`")

query = st.text_input("Nhap query", placeholder="vi du: muc phat khi vuot den do")
left_term = right_term = ""
distance = 5
if search_mode == "Classic IR Proximity":
    col1, col2, col3 = st.columns(3)
    with col1:
        left_term = st.text_input("Left term", placeholder="vi du: dat")
    with col2:
        right_term = st.text_input("Right term", placeholder="vi du: thua ke")
    with col3:
        distance = st.number_input("Distance", min_value=1, max_value=20, value=5, step=1)

search_clicked = st.button("Search", type="primary")

if search_clicked:
    if not query.strip():
        st.warning("Hay nhap query tim kiem.")
    else:
        try:
            with st.spinner("Dang tim kiem ..."):
                if search_mode == "BM25 champion":
                    engine = load_bm25_engine(model_dir)
                    results = search_bm25(engine, query=query, top_k=top_k, use_champion=True)
                elif search_mode == "BM25 full":
                    engine = load_bm25_engine(model_dir)
                    results = search_bm25(engine, query=query, top_k=top_k, use_champion=False)
                else:
                    engine = load_tfidf_engine(model_dir)
                    results = engine.search(query, top_k=top_k)
                if search_mode == "Classic IR Proximity":
                    engine = load_classic_engine(
                        bm25_model_dir=str(CLASSIC_BM25_MODEL_DIR),
                        tokenized_corpus_path=str(TOKENIZED_CORPUS_PATH),
                        positional_index_path=str(CLASSIC_INDEX_DIR / "positional_index.pkl"),
                    )
                    if not left_term.strip() or not right_term.strip():
                        st.warning("Hay nhap left term va right term cho Proximity (NEAR).")
                        st.stop()
                    results = engine.near_search(left_term, right_term, distance=int(distance), top_k=top_k)

            if search_mode == "Classic IR Proximity":
                render_classic_results(results)
            elif search_mode == "TF-IDF VSM":
                render_results(results)
            else:
                render_results(results)
        except Exception as exc:
            st.error(f"Loi khi search: {exc}")
else:
    st.info("Nhap query va bam Search de xem ket qua.")
