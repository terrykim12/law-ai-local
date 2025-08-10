# [RAG][chunker]
# 역할: TEXT_KEYS에서 본문 추출 후 정규화(BOM 제거/NFC) → chunk_size/overlap으로 분할.
# TODO:
# - 인코딩/모지바케 정규화 강화.
# - 문단/문장 경계 기반 분할 옵션(sentencize) 추가.
import argparse, json, uuid, datetime, unicodedata
from pathlib import Path
from typing import Dict, Any, Iterable, List
from src.retriever import split_text  # 기존 함수 그대로 재사용

TEXT_KEYS = ["full_text", "summary", "text", "content", "body", "judgment", "opinion", "raw_text"]
ID_KEYS   = ["case_id", "case_no", "id", "doc_id", "uid"]

def pick(d: Dict[str, Any], keys: List[str]) -> str:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""

def read_records(p: Path) -> Iterable[Dict[str, Any]]:
    s = p.read_text(encoding="utf-8", errors="ignore").lstrip("﻿")
    if s.startswith("["):
        arr = json.loads(s)
        for o in arr:
            if isinstance(o, dict): yield o
    else:
        for line in s.splitlines():
            line = line.strip()
            if not line: continue
            try:
                o = json.loads(line)
                if isinstance(o, dict): yield o
            except Exception:
                continue

def normalize(txt: str) -> str:
    # BOM 제거 + 유니코드 정규화 + 제어문자 정리
    txt = txt.lstrip("﻿")
    txt = unicodedata.normalize("NFC", txt)
    return txt.replace("\r\n", "\n").replace("\r", "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="원본 cases.jsonl 경로 (data/raw/..)")
    ap.add_argument("--chunk-size", type=int, default=800)
    ap.add_argument("--overlap", type=int, default=120)
    ap.add_argument("--outdir", default="data/processed", help="청크 파일 저장 폴더")
    args = ap.parse_args()

    src_path = Path(args.inp)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = outdir / f"chunks_{ts}.jsonl"

    total, with_text, chunks_out = 0, 0, 0
    with out_path.open("w", encoding="utf-8") as wf:
        for rec in read_records(src_path):
            total += 1
            text = pick(rec, TEXT_KEYS)
            if not text: continue
            with_text += 1
            text = normalize(text)
            case_id = pick(rec, ID_KEYS) or f"case-{uuid.uuid4().hex[:8]}"
            source  = rec.get("source") or src_path.as_posix()

            parts = split_text(text, args.chunk_size, args.overlap) or []
            for i, ch in enumerate(parts):
                ch = normalize(ch)
                # ingest에서 바로 쓸 수 있도록 'text' 키로 저장
                out = {
                    "text": ch,
                    "source": source,
                    "chunk_idx": i,
                    "doc_type": "case",
                    "case_id": case_id,
                }
                wf.write(json.dumps(out, ensure_ascii=False) + "\n")
                chunks_out += 1

    print(f"[DONE] 입력레코드 total={total}, 본문존재={with_text}")
    print(f"[DONE] 생성 청크 수={chunks_out}, 파일={out_path.as_posix()}")

if __name__ == "__main__":
    main()
