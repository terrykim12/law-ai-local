from __future__ import annotations

import csv
from pathlib import Path
from typing import List

from src.retriever import Retriever


DATA = Path("eval/examples.csv")


def evaluate() -> None:
    retriever = Retriever("config.yaml")
    rows = list(csv.DictReader(DATA.open("r", encoding="utf-8")))
    if not rows:
        print("평가 데이터가 없습니다.")
        return

    total = len(rows)
    hit = 0
    for row in rows:
        q = (row.get("question") or "").strip()
        keywords = [s.strip() for s in (row.get("keywords") or "").split(";") if s.strip()]
        chunks = retriever.query(q, top_k=5)
        context = "\n".join([c.text for c in chunks])
        ok = all(kw in context for kw in keywords) if keywords else False
        hit += 1 if ok else 0
        print(f"Q: {q}\n - Keywords: {keywords}\n - Hit: {ok}")

    print("\n=== 결과 ===")
    print(f"Count: {total}")
    print(f"Hits:  {hit}")
    print(f"Recall: {hit/total:.2%}")


if __name__ == "__main__":
    evaluate()

