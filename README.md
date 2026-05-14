# Information Retrieval - Vietnamese Legal Search

Project truy xuat thong tin tren bo du lieu Zalo AI Legal Text Retrieval VN.
Huong chay hien tai la dense retrieval voi BGE-M3 + FAISS tren full corpus.

## Cau truc chinh

```text
.
├── app.py
├── embed_bge.py
├── evaluate_bge_20k.py
├── data/
│   └── zalo_ai_legal_text_retrieval_vn/
│       ├── corpus.jsonl
│       ├── queries.jsonl
│       ├── queries_unique.jsonl
│       └── qrels/
│           ├── train.jsonl
│           └── test.jsonl
├── artifacts/
│   ├── bge_m3_legal_full/
│   └── bge_m3_queries_full/
├── pyproject.toml
└── src/
    ├── cli.py
    └── search_tfidf/
        ├── bge_m3_engine.py
        ├── champion_bm25/
        ├── documents.py
        ├── engine.py
        ├── io.py
        └── text_utils.py
```

## Du lieu

- `corpus.jsonl`: full legal corpus, 61,425 documents.
- `queries.jsonl`: query goc, co mot so query id bi lap.
- `queries_unique.jsonl`: query da loai trung theo `_id`, dung de embed/evaluate.
- `qrels/train.jsonl`: relevance labels train.
- `qrels/test.jsonl`: relevance labels test.

## Cai dat

```bash
cd  InformationRetrieval
pip install -r requirements.txt
```

Neu chay trong moi truong editable package:

```bash
pip install -e .
```

## Embed Full Corpus Bang BGE-M3

Script don gian de embed corpus:

```bash
cd /kaggle/InformationRetrieval && python embed_bge.py \
  --input data/zalo_ai_legal_text_retrieval_vn/corpus.jsonl \
  --output-dir artifacts/bge_m3_legal_full \
  --batch-size 32 \
  --max-length 1024 \
  --device cuda \
  --fp16
```

Output:

```text
artifacts/bge_m3_legal_full/bge_m3.index
artifacts/bge_m3_legal_full/bge_m3_embeddings.npy
artifacts/bge_m3_legal_full/bge_m3_meta.joblib
```

Ghi chu:

- BGE-M3 ho tro context toi khoang 8192 token.
- `--max-length 1024` nhanh va hop ly de thu nghiem full corpus.
- Neu GPU du VRAM, co the thu `--max-length 2048` hoac `4096`.
- Neu bi out-of-memory, giam `--batch-size 16` hoac `--batch-size 8`.
- `--fp16` giup giam VRAM va thuong nhanh hon tren CUDA.

## Embed Queries

Dung file query unique de tranh embed lap query:

```bash
cd /kaggle/InformationRetrieval && python embed_bge.py \
  --input data/zalo_ai_legal_text_retrieval_vn/queries_unique.jsonl \
  --output-dir artifacts/bge_m3_queries_full \
  --batch-size 32 \
  --max-length 256 \
  --device cuda \
  --fp16
```

Output:

```text
artifacts/bge_m3_queries_full/bge_m3.index
artifacts/bge_m3_queries_full/bge_m3_embeddings.npy
artifacts/bge_m3_queries_full/bge_m3_meta.joblib
```

## Evaluate Retrieval

Evaluate tren test set:

```bash
cd /kaggle/InformationRetrieval && python evaluate_bge_20k.py \
  --corpus-dir artifacts/bge_m3_legal_full \
  --query-dir artifacts/bge_m3_queries_full \
  --qrels data/zalo_ai_legal_text_retrieval_vn/qrels/test.jsonl
```

Evaluate tren train set:

```bash
cd /kaggle/InformationRetrieval && python evaluate_bge_20k.py \
  --corpus-dir artifacts/bge_m3_legal_full \
  --query-dir artifacts/bge_m3_queries_full \
  --qrels data/zalo_ai_legal_text_retrieval_vn/qrels/train.jsonl
```

Mac dinh script tinh:

```text
Recall@1,3,5,10,20,50,100
Hit@1,3,5,10,20,50,100
MRR@1,3,5,10,20,50,100
nDCG@1,3,5,10,20,50,100
```

## Search Mot Query

Neu da co index full corpus, co the search bang CLI:

```bash
cd /kaggle/InformationRetrieval && PYTHONPATH=src python -m cli search-bge \
  --model-dir artifacts/bge_m3_legal_full \
  --query "Công an xã xử phạt lỗi không mang bằng lái xe có đúng không?" \
  --top-k 5 \
  --device cuda
```

## Baseline TF-IDF/BM25

Code baseline van nam trong:

- `src/search_tfidf/engine.py`: TF-IDF + cosine similarity.
- `src/search_tfidf/bm25_engine.py`: BM25.

Neu can build baseline tren legal corpus:

```bash
PYTHONPATH=src python -m cli build-bm25 \
  --input data/zalo_ai_legal_text_retrieval_vn/corpus.jsonl \
  --model-dir artifacts/bm25_legal_full
```

## Champion BM25 Inference

Module nay chi dung cho bo Zalo AI Legal Text Retrieval VN va artifact tokenized trong `artifacts/bm25_underthesea`.

Build BM25 champion-list model mot lan tu corpus da tokenize:

```bash
python src/champion-list/champion_bm25/build_model.py --corpus-tokenized artifacts/bm25_underthesea/corpus_doc_id.jsonl --model-dir artifacts/bm25_legal_full --champion-size 9000
```

Sau do inference truc tiep bang query text:

```bash
python src/champion-list/champion_bm25/search.py --model-dir artifacts/bm25_legal_full --query "Mức phạt khi quay đầu xe ô tô trên đường cao tốc" --top-k 5
```

Luong nay hoat dong nhu sau:
- `build_model.py` build va luu `inverted_index`, `champion_index`, `idf`, `doc_lengths`, `avgdl`.
- `search.py` chi load `bm25_model.joblib`, tokenize query bang `underthesea`, roi search.
- Corpus khong tokenize lai va khong build lai chi muc luc inference.

## Evaluate Champion BM25

Neu muon do metric cua BM25 champion-list dung dung model inference da build san:

```bash
python src/champion-list/champion_bm25/evaluate.py \
  --model-dir artifacts/bm25_legal_full \
  --queries-tokenized artifacts/bm25_underthesea/queries_test_tokens.jsonl \
  --qrels data/zalo_ai_legal_text_retrieval_vn/qrels/test.jsonl \
  --mode both
```

Neu muon evaluate tren tap train:

```bash
python src/champion-list/champion_bm25/evaluate.py \
  --model-dir artifacts/bm25_legal_full \
  --queries-raw data/zalo_ai_legal_text_retrieval_vn/queries_unique.jsonl \
  --qrels data/zalo_ai_legal_text_retrieval_vn/qrels/train.jsonl \
  --mode both
```

Ghi chu:
- `--queries-tokenized` dung khi ban da co query tokenize san, hop cho benchmark latency.
- `--queries-raw` se tokenize query bang underthesea luc runtime.
- `--mode both` in ra ca BM25 full va BM25 champion list trong cung mot lan chay.
- `--champion-size` chi can o buoc build model; inference va evaluate tu `--model-dir` se dung champion size da luu san.

## Luu Y Ve Git

Khong nen commit cac file trong `artifacts/` len GitHub vi embedding/index rat lon.
GitHub chan file tren 100MB. Hay de artifacts o local/Kaggle output, hoac dung Git LFS neu that su can versioning artifact.

Nen ignore cac artifact sinh ra:

```gitignore
artifacts/bge_m3_*/
*.npy
*.index
*.joblib
```
