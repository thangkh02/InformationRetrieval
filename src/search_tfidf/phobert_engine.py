from __future__ import annotations

from pathlib import Path
from typing import Iterable

import faiss
import joblib
import numpy as np
import torch
from torch.nn.functional import normalize as l2_normalize
from transformers import AutoModel, AutoTokenizer

from .documents import NewsDocument, SearchResult
from .text_utils import prepare_phobert_text


class PhoBERTSearchEngine:
    def __init__(
        self,
        model_name: str = "vinai/phobert-base-v2",
        batch_size: int = 16,
        max_length: int = 256,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModel.from_pretrained(model_name, add_pooling_layer=False).to(self.device)
        self.model.eval()
        self.documents: list[NewsDocument] = []
        self.embeddings: np.ndarray | None = None
        self.index: faiss.Index | None = None

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoded)
            token_embeddings = outputs.last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1)
            summed = (token_embeddings * attention_mask).sum(dim=1)
            counts = attention_mask.sum(dim=1).clamp(min=1)
            embeddings = summed / counts
            embeddings = l2_normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy().astype(np.float32)

    def _encode_texts(self, texts: Iterable[str]) -> np.ndarray:
        batches = []
        current: list[str] = []
        for text in texts:
            current.append(prepare_phobert_text(text))
            if len(current) >= self.batch_size:
                batches.append(self._encode_batch(current))
                current = []
        if current:
            batches.append(self._encode_batch(current))
        if not batches:
            return np.empty((0, self.model.config.hidden_size), dtype=np.float32)
        return np.vstack(batches)

    def fit(self, documents: list[NewsDocument]) -> None:
        if not documents:
            raise ValueError("No documents provided for indexing.")

        self.documents = documents
        corpus = [doc.text for doc in documents]
        self.embeddings = self._encode_texts(corpus)
        dim = int(self.embeddings.shape[1])
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        if self.index is None or self.embeddings is None:
            raise RuntimeError("The engine has not been fitted yet.")
        if not query.strip():
            return []

        query_embedding = self._encode_texts([query])
        if query_embedding.size == 0:
            return []

        scores, indices = self.index.search(query_embedding, top_k)
        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0:
                continue
            doc = self.documents[int(idx)]
            results.append(
                SearchResult(
                    doc_id=doc.doc_id,
                    score=float(score),
                    title=doc.title,
                    summary=doc.summary,
                    category=doc.category,
                    content=doc.content,
                )
            )
        return results

    def save(self, model_dir: str | Path) -> None:
        if self.index is None or self.embeddings is None:
            raise RuntimeError("Nothing to save. Fit the engine first.")

        path = Path(model_dir)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "phobert.index"))
        np.save(path / "phobert_embeddings.npy", self.embeddings)
        joblib.dump(
            {
                "model_name": self.model_name,
                "batch_size": self.batch_size,
                "max_length": self.max_length,
                "documents": self.documents,
            },
            path / "phobert_meta.joblib",
        )

    @classmethod
    def load(cls, model_dir: str | Path, device: str | None = None) -> "PhoBERTSearchEngine":
        path = Path(model_dir)
        payload = joblib.load(path / "phobert_meta.joblib")
        engine = cls(
            model_name=payload["model_name"],
            batch_size=payload["batch_size"],
            max_length=payload["max_length"],
            device=device,
        )
        engine.documents = payload["documents"]
        engine.embeddings = np.load(path / "phobert_embeddings.npy")
        engine.index = faiss.read_index(str(path / "phobert.index"))
        return engine
