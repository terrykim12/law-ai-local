import chromadb

# 1. DB 클라이언트 연결
client = chromadb.PersistentClient(path="vectorstore")

# 2. 모든 컬렉션 이름 출력
collections = client.list_collections()
print("현재 DB에 있는 컬렉션:", [c.name for c in collections] or "(없음)")

# 3. 특정 컬렉션 내용 확인 (컬렉션 이름을 바꿔서 확인해보세요)
COLLECTION_NAME = "documents"  # 또는 "cases_kb_m3"

if not any(c.name == COLLECTION_NAME for c in collections):
    print(f"\n'{COLLECTION_NAME}' 컬렉션이 없습니다.")
else:
    col = client.get_collection(COLLECTION_NAME)
    print(f"\n--- '{COLLECTION_NAME}' 컬렉션 정보 ---")
    print("전체 청크 개수:", col.count())
    print("일부 데이터 확인:", col.peek(3))
