from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
import chromadb
from chromadb.utils import embedding_functions


def load_config(config_path: str | Path = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    end = max(chunk_size, 1)
    length = len(text)
    while start < length:
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= length:
            break
        start = max(end - chunk_overlap, 0)
        end = min(start + chunk_size, length)
    return chunks


@dataclass
class RetrievedChunk:
    id: str
    text: str
    score: Optional[float]
    metadata: Dict[str, Any]


class Retriever:
    def __init__(self, config_path: str | Path = "config.yaml") -> None:
        self.config = load_config(config_path)
        vs_cfg = self.config.get("vectorstore", {})
        embed_cfg = self.config.get("embedding", {})
        self.top_k: int = int(self.config.get("retriever", {}).get("top_k", 5))
        self.chunk_size: int = int(self.config.get("retriever", {}).get("chunk_size", 800))
        self.chunk_overlap: int = int(self.config.get("retriever", {}).get("chunk_overlap", 120))

        self.db_path = vs_cfg.get("path", "vectorstore")
        self.collection_name = "documents"
        model_name = os.getenv("EMBEDDING_MODEL", embed_cfg.get("model", "all-MiniLM-L6-v2"))

        self.client = chromadb.PersistentClient(path=self.db_path)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

        # 이미 존재하면 가져오고, 없으면 생성
        try:
            self.collection = self.client.get_collection(self.collection_name)
        except Exception:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn,
            )

    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        self.collection.add(documents=documents, metadatas=metadatas, ids=ids)

    def query(self, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        k = int(top_k or self.top_k)
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas", "distances", "ids"],
        )
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        out: List[RetrievedChunk] = []
        for doc_id, doc_text, meta, dist in zip(ids, docs, metas, dists):
            score = float(dist) if dist is not None else None
            out.append(RetrievedChunk(id=str(doc_id), text=str(doc_text), score=score, metadata=meta or {}))
        return out


__all__ = ["Retriever", "split_text", "RetrievedChunk", "load_config"]

