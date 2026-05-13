from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import joblib
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def read_corpus(path: Path) -> tuple[list[str], list[dict]]:
    texts: list[str] = []
    docs: list[dict] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)
            doc_id = item.get("_id", item.get("doc_id"))
            title = str(item.get("title", ""))
            content = str(item.get("text", item.get("content", "")))
            text = f"{title} {content}".strip()

            if not doc_id or not text:
                continue

            texts.append(text)
            docs.append(
                {
                    "doc_id": doc_id,
                    "title": title,
                    "content": content,
                }
            )

    return texts, docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed 20k legal corpus with BGE-M3.")
    parser.add_argument(
        "--input",
        default="data/zalo_ai_legal_text_retrieval_vn/corpus_20k.jsonl",
        help="Input corpus JSONL path",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/bge_m3_legal_20k",
        help="Directory to save FAISS index, embeddings, and metadata",
    )
    parser.add_argument("--model-name", default="BAAI/bge-m3", help="SentenceTransformer model name")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum token length")
    parser.add_argument("--device", default=None, help="cuda, cuda:0, or cpu")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 on CUDA to reduce VRAM")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Reading corpus: {input_path}")
    texts, docs = read_corpus(input_path)
    if not texts:
        raise ValueError(f"No valid documents found in {input_path}")

    print(f"Loaded {len(texts):,} documents")
    print(f"Loading model: {args.model_name} on {device}")
    model = SentenceTransformer(args.model_name, device=device)
    model.max_seq_length = args.max_length

    if args.fp16 and device.startswith("cuda"):
        model.half()
        print("Using fp16")

    print(f"Embedding with batch_size={args.batch_size}, max_length={args.max_length}")
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32, copy=False)

    print(f"Embeddings shape: {embeddings.shape}")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_dir / "bge_m3.index"))
    np.save(output_dir / "bge_m3_embeddings.npy", embeddings)
    joblib.dump(
        {
            "model_name": args.model_name,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "device": device,
            "documents": docs,
        },
        output_dir / "bge_m3_meta.joblib",
    )

    print("Saved:")
    print(f"- {output_dir / 'bge_m3.index'}")
    print(f"- {output_dir / 'bge_m3_embeddings.npy'}")
    print(f"- {output_dir / 'bge_m3_meta.joblib'}")


if __name__ == "__main__":
    main()
