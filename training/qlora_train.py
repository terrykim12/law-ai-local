from __future__ import annotations

"""
경량 예시 스크립트: SFT JSONL을 읽어 형식 확인 및 간단 요약을 출력합니다.
실제 QLoRA 학습 파이프라인(Transformers/PEFT/TRL 등)은 환경 의존성이 크므로 별도 구성하세요.
"""

import json
from pathlib import Path
from typing import Iterable


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    data_path = Path("training/sft.jsonl")
    if not data_path.exists():
        print("training/sft.jsonl 이 없습니다.")
        return
    items = list(read_jsonl(data_path))
    print(f"샘플 수: {len(items)}")
    if items:
        ex = items[0]
        print("예시 항목:")
        print({k: ex.get(k) for k in ("instruction", "input", "output")})


if __name__ == "__main__":
    main()

