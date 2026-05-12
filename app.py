from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from search_tfidf import TfidfSearchEngine, read_jsonl  # noqa: E402


DATA_PATH = ROOT / "data" / "news_sample.jsonl"
MODEL_DIR = ROOT / "artifacts" / "model"


st.set_page_config(page_title="IR Search", page_icon="IR", layout="wide")

st.title("IR Search")
st.caption("Tim kiem thong tin don gian bang TF-IDF + cosine similarity + top-k.")


@st.cache_resource
def load_engine() -> TfidfSearchEngine:
    model_file = MODEL_DIR / "tfidf_model.joblib"
    if model_file.exists():
        try:
            return TfidfSearchEngine.load(MODEL_DIR)
        except Exception:
            model_file.unlink(missing_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay du lieu tai {DATA_PATH}. Hay tao file sample truoc."
        )

    documents = load_documents()
    engine = TfidfSearchEngine(min_df=2 if len(documents) >= 5 else 1)
    engine.fit(documents)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    engine.save(MODEL_DIR)
    return engine


@st.cache_data
def load_documents():
    if not DATA_PATH.exists():
        return []
    return read_jsonl(DATA_PATH)


with st.sidebar:
    st.header("Thiet lap")
    top_k = st.slider("Top-k", min_value=1, max_value=20, value=5, step=1)
    st.write(f"Data: `{DATA_PATH.name}`")
    st.write(f"Model: `{MODEL_DIR.name}`")
    st.write(f"Documents: `{len(load_documents())}`")


query = st.text_input("Nhap tu khoa can tim", placeholder="vi du: kinh te Viet Nam")
search_clicked = st.button("Search", type="primary")

if search_clicked:
    if not query.strip():
        st.warning("Hay nhap tu khoa tim kiem.")
    else:
        try:
            engine = load_engine()
            results = engine.search(query, top_k=top_k)
            if not results:
                st.info("Khong tim thay ket qua phu hop.")
            else:
                st.subheader(f"Ket qua top-{len(results)}")
                for rank, result in enumerate(results, start=1):
                    with st.container(border=True):
                        st.markdown(f"**{rank}. {result.title}**")
                        st.write(f"Score: `{result.score:.4f}` | Category: `{result.category}` | Doc ID: `{result.doc_id}`")
                        if result.summary:
                            st.write(f"Summary: {result.summary}")
                        if result.content:
                            st.write(f"Content: {result.content[:500]}...")
        except Exception as exc:
            st.error(f"Loi khi search: {exc}")
else:
    st.info("Nhap query va bam Search de xem ket qua.")
