# [RAG][retriever]
# 역할: 벡터스토어(Chroma)에서 질의문과 유사한 청크 검색.
# 주의: 컬렉션명은 config.yaml의 retriever.collection_name를 우선 사용하도록 유지.
# TODO:
# - use_collection(name: str): 런타임 컬렉션 전환 지원.
# - where 필터 지원(doc_type='case' 등 메타 기반).
# - 중복 제거: 동일 source/chunk_idx 및 유사 텍스트 1개만 유지.
# - (선택) bge-reranker-large 재랭커 장착 및 top_k 조정.
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
    def __init__(
        self,
        config_path: str | Path = "config.yaml",
        collection_name: Optional[str] = None,
    ) -> None:
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
        # 외부에서 받은 collection_name을 우선 사용, 없으면 config 파일 값 사용
        self.collection_name = collection_name or retr_cfg.get("collection_name", "documents")
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

    def query(self, question: str, top_k: int = 6):
        results = self.collection.query(
            query_texts=[question],
            n_results=top_k,
            include=["documents","metadatas","distances"],
        )

        docs  = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        ids   = results.get("ids", [[]])[0]  # 없어도 안전하게 처리

        from collections import defaultdict
        hits, seen_ids, seen_sig = [], set(), set()
        file_bucket = defaultdict(list)

        for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
            _id = ids[i] if i < len(ids) else f"auto-{i}"
            if _id in seen_ids:
                continue
            sig = (meta.get("source"), meta.get("chunk_idx"), (doc or "").strip())
            if sig in seen_sig:
                continue
            seen_ids.add(_id); seen_sig.add(sig)
            sim = 1.0 / (1.0 + float(dist) if dist is not None else 1.0)
            item = {
              "id": _id,
              "text": (doc or ""),
              "score": sim,  # 0~1
              "metadata": meta or {},
              "source_id": f"{(meta or {}).get('source')}#chunk{(meta or {}).get('chunk_idx')}"
            }
            file_bucket[(meta or {}).get("source")].append(item)

        balanced = []
        for _, lst in file_bucket.items():
            balanced.extend(lst[:2])  # 파일당 최대 2개
        return balanced

    def _get_reranker(self) -> CrossEncoder:
        if self._reranker is None:
            model_name = self.reranker_model or "BAAI/bge-reranker-large"
            # embedder와 동일 디바이스 선호
            device = self.embedding_fn._model.device.type if hasattr(self.embedding_fn, "_model") else "cpu"  # type: ignore[attr-defined]
            self._reranker = CrossEncoder(model_name, device=device)
        return self._reranker


__all__ = ["Retriever", "split_text", "RetrievedChunk", "load_config"]

