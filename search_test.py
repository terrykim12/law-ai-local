from src.retriever import Retriever
import argparse

# --- 스크립트 실행 시 컬렉션 이름을 지정할 수 있도록 기능 추가 ---
parser = argparse.ArgumentParser(description="검색할 컬렉션 이름.")
parser.add_argument(
    "-c", "--collection", 
    default=None, 
    help="검색할 컬렉션 이름. (기본값: config.yaml에 지정된 컬렉션)"
)
args = parser.parse_args()


# 1. Retriever 불러오기
# 실행 시 입력한 컬렉션 이름으로 Retriever를 초기화합니다.
# 입력이 없으면 config.yaml의 기본값을 사용합니다.
r = Retriever(collection_name=args.collection)


# 2. 검색어 입력
query = "교통사고 손해배상 판례"
print(f"질의: \"{query}\"")
print(f"'{r.collection_name}' 컬렉션에서 검색합니다...")

results = r.query(query, top_k=3)

# 3. 출력
if not results:
    print("검색 결과가 없습니다.")

for i, doc in enumerate(results, 1):
    print(f"\n=== 결과 {i} (유사도: {doc.get('score', 0):.4f}) ===")
    print("본문:", (doc.get('text') or '')[:200], "...")
    print("메타데이터:", doc.get('metadata'))

print("\n---")
print("팁: 판례 컬렉션을 테스트하려면 아래처럼 실행하세요:")
print("python search_test.py --collection cases_kb_m3")
print("---")