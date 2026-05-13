# Zalo TF-IDF + Vector Space Model

## 1) Build index
```bash
python code/zalo_tfidf_vsm/run_zalo_tfidf_vsm.py build ^
  --corpus data/zalo_ai_legal_text_retrieval_vn/corpus.jsonl ^
  --output-dir artifacts/zalo_tfidf_vsm
```

## 2) Search thử
```bash
python code/zalo_tfidf_vsm/run_zalo_tfidf_vsm.py search ^
  --index artifacts/zalo_tfidf_vsm/zalo_tfidf_vsm.joblib ^
  --query "Công an xã có được xử phạt không mang bằng lái xe không?" ^
  --top-k 5
```

## 3) Đánh giá trên test qrels
```bash
python code/zalo_tfidf_vsm/run_zalo_tfidf_vsm.py eval ^
  --index artifacts/zalo_tfidf_vsm/zalo_tfidf_vsm.joblib ^
  --queries data/zalo_ai_legal_text_retrieval_vn/queries.jsonl ^
  --qrels data/zalo_ai_legal_text_retrieval_vn/qrels/test.jsonl ^
  --k 10
```

Ghi chú:
- Dữ liệu Zalo trong repo có thể bị lỗi hiển thị tiếng Việt trên PowerShell, nhưng script đọc `utf-8` nên vẫn chạy bình thường.
- Có thể tăng `--max-features` hoặc `--ngram-max` để cải thiện chất lượng retrieval.
