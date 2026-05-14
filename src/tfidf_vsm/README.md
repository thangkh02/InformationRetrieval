# TF-IDF + Vector Space Model

> **Pure Python** — không dùng `sklearn`, `numpy`, hay `joblib`.  
> Chỉ dùng stdlib: `math`, `re`, `unicodedata`, `collections`, `pickle`.

---

## 1) Build index
```bash
python src/tfidf_vsm/run_tfidf_vsm.py build ^
  --corpus data/zalo_ai_legal_text_retrieval_vn/corpus.jsonl ^
  --output-dir artifacts/tfidf_vsm ^
  --max-features 200000 ^
  --min-df 1 ^
  --ngram-max 2
```
Kết quả: `artifacts/tfidf_vsm/tfidf_vsm.pkl`

---

## 2) Search thử
```bash
python src/tfidf_vsm/run_tfidf_vsm.py search ^
  --index artifacts/tfidf_vsm ^
  --query "Công an xã có được xử phạt không mang bằng lái xe không?" ^
  --top-k 5
```

---

## 3) Đánh giá trên test qrels
```bash
python src/tfidf_vsm/run_tfidf_vsm.py eval ^
  --index artifacts/tfidf_vsm ^
  --queries data/zalo_ai_legal_text_retrieval_vn/queries.jsonl ^
  --qrels data/zalo_ai_legal_text_retrieval_vn/qrels/test.jsonl ^
  --k 10
```

---

## Ghi chú
- `--index` nhận **thư mục** chứa file `tfidf_vsm.pkl` (không phải đường dẫn file).
- Có thể tăng `--max-features` hoặc `--ngram-max` để cải thiện chất lượng retrieval.
- Dữ liệu tiếng Việt trên PowerShell có thể lỗi font, nhưng script đọc `utf-8` nên vẫn chạy bình thường.
- Cũng có thể dùng `cli_tfidf.py` (dùng `TfidfVSMEngine` từ `engine.py`) với cùng logic.
