from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., description="사용자 질문")
    top_k: int = Field(5, ge=1, le=50, description="검색 상위 K")


class Source(BaseModel):
    id: str
    text: str
    score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]


class IngestStats(BaseModel):
    num_documents: int
    num_chunks: int

