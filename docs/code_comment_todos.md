# 법률 RAG 프로젝트 — 주석/TODO 가이드 (Gemini CLI 편집용)

아래 블록을 각 파일 상단(또는 함수 위)에 붙여 넣으면, 협업자가 의도를 쉽게 파악할 수 있습니다.

- retriever.py: 컬렉션 전환(use_collection), where 필터, 중복 제거, reranker
- ingest_cases.py: 중복 스킵(증분), near-dup 제거, 배치 add/진행률, 실패 라인 로그
- chunk_cases.py: 인코딩/정규화 강화, 문장 경계 분할 옵션
- llm.py: CoT 억제, 섹션 포맷 고정, JSON 스키마 검증
- app/api/ask/route.ts: 프록시 안정화, .env 분리, 에러/지연 로깅
- UI 컴포넌트: 출처 표시/중복 collapse/UX 향상