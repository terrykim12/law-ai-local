## law-ai-local

경량 로컬 법률 RAG 실험/데모 프로젝트입니다. 문서를 로컬 벡터DB에 색인하고, 검색된 근거로 LLM(Ollama)을 호출해 답변을 생성합니다.

### 구성 요소
- 데이터 적재: `src/ingest.py` (PDF/TXT → 청킹 → 임베딩 → ChromaDB 색인)
- 검색/재랭킹: `src/retriever.py`
- LLM 호출 래퍼: `src/llm.py` (기본: Ollama)
- API 서버: `src/server.py` (FastAPI)
- 스키마: `src/schemas.py`
- 안전 가드: `src/guardrails/safety.py`
- 평가: `eval/evaluate.py`
- 학습 예시(QLoRA): `training/qlora_train.py`, `training/sft.jsonl`

### 요구사항
- Python 3.10+
- (선택) Ollama 설치 및 모델 풀링: `llama3` 등

### 설치
```bash
python -m venv .venv
./.venv/Scripts/Activate.ps1  # Windows PowerShell
pip install -U pip
pip install -r requirements.txt
```

### 설정
1) `config.yaml`에서 임베딩/LLM/검색 설정을 확인/수정합니다.
2) `.env.example` 또는 `env.example`를 참고해 `.env`를 생성합니다(필요 시 OpenAI 키 등).

### 데이터 적재(인덱싱)
1) 원본 문서를 `data/raw/`에 넣습니다(PDF/TXT 지원).
2) 아래 명령으로 색인합니다.
```bash
python -m src.ingest
```
완료 후 `vectorstore/`에 로컬 DB가 생성되고, `data/processed/`에 청킹 결과 jsonl이 저장됩니다.

### 서버 실행
```bash
uvicorn src.server:app --host 0.0.0.0 --port 8000
```

### 예시 호출
```bash
curl -X POST "http://localhost:8000/query" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"계약 해제의 요건은?\",\"top_k\":5}"
```

### 평가
`eval/examples.csv`를 수정한 뒤:
```bash
python -m eval.evaluate
```

### 학습(예시)
`training/sft.jsonl` 형식으로 데이터를 준비합니다. QLoRA 파이프라인은 환경/리소스 의존성이 커서 본 저장소에는 경량 예시만 포함했습니다. 필요 시 `training/qlora_train.py`를 참고해 맞춤 구현을 확장하세요.


