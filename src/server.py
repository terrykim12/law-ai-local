from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import FastAPI

from .schemas import QueryRequest, QueryResponse, Source
from .retriever import Retriever, load_config
from .llm import LLMClient
from .guardrails.safety import add_disclaimer, moderate_text


app = FastAPI(title="law-ai-local")

CONFIG = load_config("config.yaml")
SYSTEM_PROMPT_PATH = Path(CONFIG.get("prompts", {}).get("system", "prompts/system.txt"))
SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8") if SYSTEM_PROMPT_PATH.exists() else ""

RETRIEVER = Retriever("config.yaml")
LLM = LLMClient("config.yaml")


def build_prompt(context: str, question: str) -> str:
    return (
        "다음 맥락을 활용해 한국어로 간결하고 정확하게 답하세요.\n"
        "맥락에 근거한 부분은 번호로 근거를 제시하세요.\n\n"
        f"[맥락]\n{context}\n\n[질문]\n{question}\n\n[답변]"
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    retrieved = RETRIEVER.query(req.question, top_k=req.top_k)
    sources: List[Source] = []
    context_parts: List[str] = []
    for r in retrieved:
        sources.append(Source(id=r.id, text=r.text, score=r.score, metadata=r.metadata))
        src = r.metadata.get("source", "unknown")
        context_parts.append(f"- ({src})\n{r.text}")
    context = "\n\n".join(context_parts)

    prompt = build_prompt(context=context, question=req.question)
    # 요청에서 모델 오버라이드가 있으면 적용 (예: 'qwen3:8b')
    model_override = req.model
    raw_answer = LLM.generate(prompt=prompt, system=SYSTEM_PROMPT, model_override=model_override)
    moderated_answer = moderate_text(raw_answer)
    final_answer = add_disclaimer(moderated_answer)
    return QueryResponse(answer=final_answer, sources=sources)


__all__ = ["app"]

