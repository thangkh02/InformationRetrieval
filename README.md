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
- `src/cli.py`: lenh command line de build model va search.
- `src/search_tfidf/__init__.py`: gom export cac thanh phan chinh cua module search.
- `src/search_tfidf/documents.py`: dinh nghia model du lieu `NewsDocument` va `SearchResult`.
- `src/search_tfidf/io.py`: doc/ghi du lieu JSONL.
- `src/search_tfidf/engine.py`: logic TF-IDF, cosine similarity, va xep hang top-k.
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
```

