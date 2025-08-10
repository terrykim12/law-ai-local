from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import yaml
from pypdf import PdfReader

from .retriever import Retriever, split_text, load_config


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    texts: List[str] = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            texts.append("")
    return "\n".join(texts)


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def collect_documents(raw_dir: Path) -> List[Dict]:
    docs: List[Dict] = []
    for p in raw_dir.rglob("*"):
        if not p.is_file():
            continue
        text = ""
        if p.suffix.lower() == ".pdf":
            text = read_pdf(p)
        elif p.suffix.lower() in {".txt", ".md"}:
            text = read_txt(p)
        else:
            continue
        docs.append({"source": str(p), "text": text})
    return docs


def main() -> None:
    config = load_config("config.yaml")
    chunk_size = int(config.get("retriever", {}).get("chunk_size", 800))
    chunk_overlap = int(config.get("retriever", {}).get("chunk_overlap", 120))

    retriever = Retriever("config.yaml")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    docs = collect_documents(RAW_DIR)
    if not docs:
        print("data/raw/ 에 파일이 없습니다. PDF/TXT 파일을 추가한 뒤 다시 실행하세요.")
        return

    all_chunks: List[str] = []
    all_metas: List[Dict] = []

    for doc in docs:
        text = doc["text"]
        source = doc["source"]
        chunks = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for idx, ch in enumerate(chunks):
            all_chunks.append(ch)
            all_metas.append({"source": source, "chunk_idx": idx})

    ids = [str(uuid.uuid4()) for _ in all_chunks]
    retriever.add_documents(documents=all_chunks, metadatas=all_metas, ids=ids)

    # processed 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = PROCESSED_DIR / f"chunks_{timestamp}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for ident, text, meta in zip(ids, all_chunks, all_metas):
            f.write(json.dumps({"id": ident, "text": text, "metadata": meta}, ensure_ascii=False) + "\n")

    print(f"문서 수: {len(docs)}, 청크 수: {len(all_chunks)}")
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()

