from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import faiss
import joblib
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

CURRENT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = CURRENT_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from documents import NewsDocument, SearchResult


class BGEM3SearchEngine:
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 8,
        max_length: int = 1024,
        device: str | None = None,
        index_type: str = "flat",
        nlist: int = 100,
        nprobe: int = 10,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.max_seq_length = max_length
        self.documents: list[NewsDocument] = []
        self.embeddings: np.ndarray | None = None
        self.index: faiss.Index | None = None

    def _encode_texts(self, texts: Iterable[str]) -> np.ndarray:
        prepared = [text.strip() for text in texts if text and text.strip()]
        if not prepared:
            return np.empty((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)

        embeddings = self.model.encode(
            prepared,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return embeddings.astype(np.float32, copy=False)

    def fit(self, documents: list[NewsDocument]) -> None:
        if not documents:
            raise ValueError("No documents provided for indexing.")

        self.documents = documents
        corpus = [doc.text for doc in documents]
        self.embeddings = self._encode_texts(corpus)
        dim = int(self.embeddings.shape[1])
        self.index = self._create_index(dim, self.embeddings)
        self.index.add(self.embeddings)

    def _create_index(self, dim: int, embeddings: np.ndarray) -> faiss.Index:
        if self.index_type == "flat":
            return faiss.IndexFlatIP(dim)

        if self.index_type == "ivf_flat":
            actual_nlist = max(1, min(self.nlist, embeddings.shape[0]))
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, actual_nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(embeddings)
            index.nprobe = max(1, min(self.nprobe, actual_nlist))
            return index

        raise ValueError(f"Unsupported index_type: {self.index_type}")

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
        faiss.write_index(self.index, str(path / "bge_m3.index"))
        np.save(path / "bge_m3_embeddings.npy", self.embeddings)
        joblib.dump(
            {
                "model_name": self.model_name,
                "batch_size": self.batch_size,
                "max_length": self.max_length,
                "index_type": self.index_type,
                "nlist": self.nlist,
                "nprobe": self.nprobe,
                "documents": self.documents,
            },
            path / "bge_m3_meta.joblib",
        )

    @classmethod
    def load(cls, model_dir: str | Path, device: str | None = None) -> "BGEM3SearchEngine":
        path = Path(model_dir)
        payload = joblib.load(path / "bge_m3_meta.joblib")
        engine = cls(
            model_name=payload["model_name"],
            batch_size=payload["batch_size"],
            max_length=payload["max_length"],
            device=device,
            index_type=payload.get("index_type", "flat"),
            nlist=payload.get("nlist", 100),
            nprobe=payload.get("nprobe", 10),
        )
        engine.documents = [
            doc
            if isinstance(doc, NewsDocument)
            else NewsDocument(
                doc_id=doc.get("doc_id", doc.get("_id", "")),
                title=str(doc.get("title", "")),
                summary=str(doc.get("summary", "")),
                category=str(doc.get("category", "")),
                content=str(doc.get("content", doc.get("text", ""))),
            )
            for doc in payload["documents"]
        ]
        engine.embeddings = np.load(path / "bge_m3_embeddings.npy")
        engine.index = faiss.read_index(str(path / "bge_m3.index"))
        return engine
