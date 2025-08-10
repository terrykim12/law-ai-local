from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder


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
        # 새 키(`embedder`) 우선, 구버전(`embedding`) 호환
        embed_cfg = self.config.get("embedder", self.config.get("embedding", {}))
        retr_cfg = self.config.get("retriever", {})

        self.top_k: int = int(retr_cfg.get("top_k", 5))
        self.chunk_size: int = int(self.config.get("retriever", {}).get("chunk_size", 800))
        self.chunk_overlap: int = int(self.config.get("retriever", {}).get("chunk_overlap", 120))
        self.use_reranker: bool = bool(retr_cfg.get("use_reranker", False))
        self.reranker_model: str | None = retr_cfg.get("reranker_model")

        self.db_path = vs_cfg.get("path", "vectorstore")
        self.collection_name = "documents"
        model_name = os.getenv("EMBEDDING_MODEL", embed_cfg.get("model", "all-MiniLM-L6-v2"))
        device = str(embed_cfg.get("device", "cpu"))

        self.client = chromadb.PersistentClient(path=self.db_path)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name,
            device=device,
        )

        # 이미 존재하면 가져오고, 없으면 생성
        try:
            # chromadb 최신 버전은 get_collection에도 embedding_function 전달 가능
            self.collection = self.client.get_collection(
                self.collection_name,
                embedding_function=self.embedding_fn,  # type: ignore[call-arg]
            )
        except Exception:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn,
            )

        self._reranker: CrossEncoder | None = None

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

        # 선택적 CrossEncoder 재랭킹
        if self.use_reranker and out:
            reranker = self._get_reranker()
            pairs = [(query, item.text) for item in out]
            try:
                scores = reranker.predict(pairs)  # type: ignore[assignment]
                # 높은 점수가 상위로 오도록 정렬
                scored = list(zip(out, scores))
                scored.sort(key=lambda x: float(x[1]), reverse=True)
                out = [RetrievedChunk(
                    id=it.id,
                    text=it.text,
                    score=float(sc),
                    metadata={**(it.metadata or {}), "reranked": True},
                ) for it, sc in scored[:k]]
            except Exception:
                # 재랭커 실패 시 원 순서 유지
                pass

        return out

    def _get_reranker(self) -> CrossEncoder:
        if self._reranker is None:
            model_name = self.reranker_model or "BAAI/bge-reranker-large"
            # embedder와 동일 디바이스 선호
            device = self.embedding_fn._model.device.type if hasattr(self.embedding_fn, "_model") else "cpu"  # type: ignore[attr-defined]
            self._reranker = CrossEncoder(model_name, device=device)
        return self._reranker


__all__ = ["Retriever", "split_text", "RetrievedChunk", "load_config"]

