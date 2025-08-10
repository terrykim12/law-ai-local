# [RAG][generation]
# 역할: 시스템 프롬프트 + 컨텍스트로 모델 호출, '요지/핵심 법리/원문 인용/해석/면책' 포맷 생성.
# TODO:
# - 내부사고 출력 금지: stop 토큰 + strip_think 필수 적용.
# - (선택) JSON 스키마 강제 및 검증.
# - (선택) 인용문 자동 샘플링 + 출처 바인딩 강화.
import os
import requests
import re
from typing import List, Dict, Any

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

SYS = (
    "너는 법률·규정 기반 RAG 어시스턴트다. 내부 사고/분석 과정은 절대 출력하지 마라.\n"
    "제공된 참고 자료 범위 안에서만 답하고, 가능하면 원문 1~2줄을 인용하라.\n"
    "출처는 [파일명#chunk번호] 형식으로 표기한다.\n"
    "자료에 없으면 그 사실을 말하고 필요한 문서를 제안한다. 마지막 줄에 면책 문구를 붙인다."
)

CHAT_OPTIONS = {
    "temperature": 0.2,
    "num_ctx": 4096,
    "keep_alive": "5m",
    # ⛔️ think 관련 stop 제거합니다. 정답이 잘립니다.
    # "stop": ["</think>", "<think>", "<|assistant_thought|>", "```thinking"]
}

THINK_PATTERNS = [
    (r"<think>.*?</think>", re.S | re.I),
    (r"<\|assistant_thought\|.*?(?:<\|assistant\|>|$)", re.S | re.I),
    (r"```thinking.*?```", re.S | re.I),
    # 모델이 실수로 JSON을 그대로 내보낼 때 방어
    (r"\"sources\"\s*:\s*\[.*?\]", re.S | re.I),
    (r"\"metadata\"\s*:\s*\{.*?\}", re.S | re.I),
]

def _strip_think(text: str) -> str:
    if not text:
        return text
    for pat, flags in THINK_PATTERNS:
        text = re.sub(pat, "", text, flags=flags)
    return text.strip()

def _chat(model: str, prompt: str) -> str:
    payload = {
        "model": model,
        "stream": False,
        "options": CHAT_OPTIONS,
        "messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": prompt + "\n\n형식: 요지 → (필요시) 원문 인용 → 출처 표기"}
        ]
    }
    r = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    return ((data.get("message") or {}).get("content") or "").strip()

def _generate(model: str, prompt: str) -> str:
    payload = {
        "model": model,
        "prompt": SYS + "\n\n" + prompt + "\n\n형식: 요지 → (필요시) 원문 인용 → 출처 표기",
        "stream": False,
        "options": CHAT_OPTIONS
    }
    r = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()

def build_context(chunks: List[Dict[str, Any]]) -> str:
    # 딕셔너리 전체가 아니라 텍스트만, 사람이 읽을 수 있는 출처 표기로 구성
    parts = []
    for c in chunks:
        src = c.get("source_id") or f"{(c.get('metadata') or {}).get('source')}#chunk{(c.get('metadata') or {}).get('chunk_idx')}"
        txt = (c.get("text") or "").strip()
        if not txt:
            continue
        parts.append(f"[{src}]\n{txt}")
    return "\n\n".join(parts)

def answer_question(question: str, retrieved_chunks: List[Dict[str, Any]], model_name: str) -> str:
    context = build_context(retrieved_chunks)
    prompt = (
        f"질문: {question}\n\n"
        f"아래 참고 자료만 사용하여 답하라:\n{context if context else '(참고 자료 없음)'}"
    )

    # 1차 시도 (chat)
    try:
        ans = _chat(model_name, prompt)
    except Exception:
        ans = ""

    # 2차 시도 (generate)
    if not ans:
        try:
            ans = _generate(model_name, prompt)
        except Exception:
            ans = ""

    # 3차 시도: instruct 변형
    if not ans and not model_name.endswith("-instruct"):
        try_model = model_name + "-instruct"
        try:
            ans = _chat(try_model, prompt) or _generate(try_model, prompt)
        except Exception:
            pass

    ans = _strip_think(ans or "")
    return ans if ans else "자료에 없음 또는 모델 응답이 비었습니다."
