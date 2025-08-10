# [RAG][ingest]
# 역할: 원본 케이스(JSON/JSONL)에서 텍스트 추출 → 청크 → Chroma upsert.
# TODO:
# - existing-ID 조회로 증분 인덱싱(중복 스킵).
# - SimHash/임베딩 유사도 기반 near-duplicate 제거.
# - 배치 add + 진행률 로그 + --max N 옵션.
# - 실패 라인 로그 파일로 저장(skipped_lines.log).
import argparse, json, uuid, hashlib, itertools
from pathlib import Path
from typing import Dict, Any, Iterable, List
from src.retriever import Retriever, split_text

TEXT_KEYS = ["text", "content", "body", "judgment", "opinion", "raw_text", "full_text", "summary"]
ID_KEYS   = ["case_id", "id", "doc_id", "uid", "case_no"]

def get_existing_ids(r: Retriever, ids: List[str], batch: int = 512) -> set[str]:
    """ID 목록을 받아 DB에서 이미 존재하는 ID들을 확인합니다."""
    exist = set()
    for i in range(0, len(ids), batch):
        part = ids[i:i+batch]
        try:
            # ID만 조회하여 네트워크 부하 최소화
            got = r.collection.get(ids=part, include=[])  # type: ignore
            for g in got.get("ids", []):
                exist.add(g)
        except Exception:
            # DB에 따라 오류가 날 수 있으나, 전체를 중단시키진 않음
            pass
    return exist

def pick(d: Dict[str, Any], keys: List[str]) -> str:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

def read_records(p: Path) -> Iterable[Dict[str, Any]]:
    """JSONL도, JSON 배열 파일도 모두 지원"""
    s = p.read_text(encoding="utf-8", errors="ignore").lstrip()
    if s.startswith("["):
        arr = json.loads(s)
        for obj in arr:
            if isinstance(obj, dict):
                yield obj
    else:
        for line in s.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj

def short_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()[:10]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--collection", default=None, help="override collection name")
    args = ap.parse_args()

    # Retriever 준비
    r = Retriever(args.config, collection_name=args.collection)

    p = Path(args.path)
    if not p.exists():
        raise FileNotFoundError(p)

    docs, metas, ids = [], [], []
    total, used, skipped_no_text = 0, 0, 0

    print(f"[INFO] loading from {p.as_posix()} ...")
    for obj in read_records(p):
        total += 1
        text = pick(obj, TEXT_KEYS)
        if not text:
            skipped_no_text += 1
            continue

        case_id = pick(obj, ID_KEYS) or f"case-{uuid.uuid4().hex[:8]}"
        source  = obj.get("source") or p.as_posix()
        record_uid = uuid.uuid4().hex # Generate a unique ID for each record

        chunks = split_text(text, r.chunk_size, r.chunk_overlap) or []
        base = f"{case_id}:{record_uid}" # Use the unique record ID in the base
        for i, ch in enumerate(chunks):
            docs.append(ch)
            metas.append({
                "source": source,
                "chunk_idx": i,
                "doc_type": "case",
                "case_id": case_id,
            })
            ids.append(f"{base}#c{i}")
        used += 1

    print(f"[INFO] records total={total}, with_text={used}, no_text={skipped_no_text}")
    print(f"[INFO] chunks to add={len(docs)} (chunk_size={r.chunk_size}, overlap={r.chunk_overlap})")

    if not docs:
        print("[HINT] 본문 키 후보:", TEXT_KEYS)
        print("[HINT] 첫 몇 줄을 확인해 보세요: PowerShell ⇒  Get-Content data\raw\cases.jsonl -TotalCount 3")
        raise SystemExit("추출된 청크가 없습니다. 데이터 키 이름을 확인하세요.")

    # --- 증분 인덱싱 로직 ---
    existing = get_existing_ids(r, ids)
    if existing:
        print(f"[INFO] 이미 존재하는 ID {len(existing)}건은 건너뜁니다.")

    flt_docs, flt_metas, flt_ids = [], [], []
    for d, m, _id in zip(docs, metas, ids):
        if _id in existing:
            continue
        flt_docs.append(d); flt_metas.append(m); flt_ids.append(_id)

    if not flt_docs:
        print("[INFO] 추가할 신규 청크가 없습니다.")
        return

    r.add_documents(flt_docs, flt_metas, flt_ids)
    print(f"[DONE] 신규 {len(flt_docs)}개 청크 추가")
    
if __name__ == "__main__":
    main()
