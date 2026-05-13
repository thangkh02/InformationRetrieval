# IR Search

Project tim kiem thong tin don gian, tap trung vao TF-IDF top-k.

## Cau truc du an

```text
.
├── app.py
├── data/
│   └── news_sample.jsonl
├── artifacts/
│   └── model/
├── pyproject.toml
└── src/
    ├── cli.py
    └── search_tfidf/
        ├── __init__.py
        ├── documents.py
        ├── engine.py
        └── io.py
```

## Mo ta tung file

- `app.py`: giao dien Streamlit de nhap query va xem ket qua top-k.
- `src/cli.py`: lenh command line de build model va search, bao gom TF-IDF va BM25.
- `src/search_tfidf/__init__.py`: gom export cac thanh phan chinh cua module search.
- `src/search_tfidf/documents.py`: dinh nghia model du lieu `NewsDocument` va `SearchResult`.
- `src/search_tfidf/io.py`: doc/ghi du lieu JSONL.
- `src/search_tfidf/engine.py`: logic TF-IDF, cosine similarity, va xep hang top-k.
- `src/search_tfidf/bm25_engine.py`: BM25 retrieval co chuan hoa text va top-k ranking.
- `src/search_tfidf/text_utils.py`: chuan hoa text chung cho lowercasing, bo dau va loai ky tu dac biet.
- `data/news_sample.jsonl`: tap 10k ban ghi da lay mau tu dataset goc.
- `artifacts/model/`: noi luu model TF-IDF sau khi build.

## Pham vi hien tai

- Xay TF-IDF index tu `data/news_sample.jsonl`
- Tim kiem bang cosine similarity
- Tra ve top-k ket qua
- Hien thi ket qua qua Streamlit hoac CLI

## Chay project

```bash
pip install -e .
streamlit run app.py
```

## Chay CLI

```bash
ir-search build --input data/news_sample.jsonl --model-dir artifacts/model
ir-search search --model-dir artifacts/model --query "kinh te Viet Nam" --top-k 5
ir-search build-bm25 --input data/news_sample.jsonl --model-dir artifacts/model
ir-search search-bm25 --model-dir artifacts/model --query "kinh te Viet Nam" --top-k 5
```

## Build vector index BGE-M3 cho legal corpus

```bash
PYTHONPATH=src python -m cli build-bge \
  --input data/zalo_ai_legal_text_retrieval_vn/corpus.jsonl \
  --model-dir artifacts/bge_m3_legal \
  --batch-size 4 \
  --max-length 1024
```

Sau khi build xong, vector va FAISS index se nam trong:

```text
artifacts/bge_m3_legal/bge_m3.index
artifacts/bge_m3_legal/bge_m3_embeddings.npy
artifacts/bge_m3_legal/bge_m3_meta.joblib
```

Search bang BGE-M3:

```bash
PYTHONPATH=src python -m cli search-bge \
  --model-dir artifacts/bge_m3_legal \
  --query "Công an xã xử phạt lỗi không mang bằng lái xe có đúng không?" \
  --top-k 5
```
