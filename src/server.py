import os
from fastapi import FastAPI, HTTPException
from .schemas import QueryRequest, QueryResponse
from .retriever import Retriever
from .llm import answer_question
from pydantic import BaseModel

app = FastAPI()

RETRIEVER = Retriever("config.yaml")
CASES_RETRIEVER = Retriever(config_path="config.yaml", collection_name="cases_kb_m3")

class AskCasesRequest(BaseModel):
    question: str
    model: str | None = None

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    try:
        ctx = RETRIEVER.query(req.question, top_k=req.top_k or 6)

        # 모델명 미입력 시 config 기본값(LLM_DEFAULT)로 폴백
        model_name = (req.model or os.environ.get("LLM_DEFAULT") or "qwen2.5:7b-instruct")
        ans = answer_question(req.question, ctx, model_name)

        return {
            "answer": ans,
            "sources": ctx,
        }
    except Exception as e:
        # 콘솔에 전체 스택을 찍고 사용자에겐 간단 메시지
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server error: {e}")

@app.post("/ask_cases", response_model=QueryResponse)
def ask_cases(req: AskCasesRequest):
    try:
        # 판례 컬렉션에서만 검색
        ctx = CASES_RETRIEVER.query(req.question, top_k=6)

        # 모델명 미입력 시 config 기본값 또는 환경변수로 폴백
        model_name = (req.model or os.environ.get("LLM_DEFAULT") or "qwen3:8b")
        ans = answer_question(req.question, ctx, model_name=model_name)

        return {
            "answer": ans,
            "sources": ctx,
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server error: {e}")
