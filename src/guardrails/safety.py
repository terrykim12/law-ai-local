from __future__ import annotations

import re


DISCLAIMER = (
    "\n\n[면책] 본 답변은 일반적 정보 제공 목적이며, 구체 사안에 대한 법률 자문이 아닙니다. "
    "사안에 따라 결과가 달라질 수 있으므로, 전문가와 상담을 권장합니다."
)


def add_disclaimer(answer: str) -> str:
    return (answer or "").rstrip() + DISCLAIMER


def moderate_text(text: str) -> str:
    # 매우 단순한 금칙어 필터(데모 용). 실제 환경에선 전문 모더레이션 모델/룰을 사용하세요.
    banned = ["불법", "사기", "악성코드", "해킹"]
    cleaned = text
    for w in banned:
        cleaned = re.sub(re.escape(w), "**[비공개]**", cleaned, flags=re.IGNORECASE)
    return cleaned


__all__ = ["add_disclaimer", "moderate_text"]

