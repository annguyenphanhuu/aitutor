"""
Script ingest dữ liệu RAG vào ChromaDB.

Kiến trúc:
  math_theory  ← data/theory.json + data/formulas.json

Exam data (data/exams/*.json) KHÔNG được ingest vào vector DB.
Chúng chỉ dùng cho quiz/evaluation qua JSON, tránh rò rỉ đề vào RAG context.

Cách dùng:
  python scripts/ingest.py               # ingest tất cả (theory + formulas)
  python scripts/ingest.py --theory      # chỉ ingest lý thuyết
  python scripts/ingest.py --formulas    # chỉ ingest công thức
  python scripts/ingest.py --from-s3     # pull từ S3 rồi ingest tất cả
"""

import json
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag.knowledge_base import get_knowledge_base

DATA_DIR       = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
THEORY_FILE    = os.path.join(DATA_DIR, "theory.json")
FORMULAS_FILE  = os.path.join(DATA_DIR, "formulas.json")
VALID_TYPES    = {"theory", "formula"}


def load_json(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def validate_docs(docs: list[dict], source: str) -> list[dict]:
    """Validate và lọc bỏ các doc lỗi, trả về danh sách hợp lệ."""
    valid = []
    for doc in docs:
        doc_id = doc.get("id", "?")

        if not doc.get("id"):
            print(f"  [SKIP] Thiếu 'id' trong {source}")
            continue
        if not doc.get("content"):
            print(f"  [SKIP] [{doc_id}] Thiếu 'content'")
            continue

        meta = doc.get("metadata", {})
        dtype = meta.get("type", "")
        if dtype not in VALID_TYPES:
            print(f"  [SKIP] [{doc_id}] type='{dtype}' không hợp lệ (chỉ nhận: {VALID_TYPES})")
            continue

        valid.append(doc)
    return valid


def ingest_theory(kb) -> int:
    """Ingest theory.json vào collection math_theory."""
    print("\n--- INGEST LÝ THUYẾT ---")
    print(f"  Nguồn: {THEORY_FILE}")

    if not os.path.exists(THEORY_FILE):
        print("  File theory.json chưa có, bỏ qua.")
        return 0

    docs  = load_json(THEORY_FILE)
    valid = validate_docs(docs, "theory.json")

    theory_docs = [d for d in valid if d["metadata"]["type"] == "theory"]
    if not theory_docs:
        print("  Không có document nào hợp lệ.")
        return 0

    kb.ingest_theory(theory_docs)
    print(f"  OK: {len(theory_docs)} documents -> math_theory")
    return len(theory_docs)


def ingest_formulas(kb) -> int:
    """Ingest formulas.json vào collection math_theory (cùng collection)."""
    print("\n--- INGEST CÔNG THỨC ---")
    print(f"  Nguồn: {FORMULAS_FILE}")

    if not os.path.exists(FORMULAS_FILE):
        print("  File formulas.json chưa có, bỏ qua.")
        return 0

    docs  = load_json(FORMULAS_FILE)
    valid = validate_docs(docs, "formulas.json")

    formula_docs = [d for d in valid if d["metadata"]["type"] == "formula"]
    if not formula_docs:
        print("  Không có formula nào hợp lệ.")
        return 0

    kb.ingest_theory(formula_docs)
    print(f"  OK: {len(formula_docs)} formulas -> math_theory")
    return len(formula_docs)


def main():
    parser = argparse.ArgumentParser(description="Ingest RAG data vào ChromaDB")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--theory",   action="store_true", help="Chỉ ingest lý thuyết")
    group.add_argument("--formulas", action="store_true", help="Chỉ ingest công thức")
    group.add_argument("--from-s3",  action="store_true", help="Pull từ S3 rồi ingest tất cả")
    args = parser.parse_args()

    print("AITutor RAG Ingest")
    print("=" * 50)

    # ── Pull từ S3 trước nếu có flag --from-s3 ────────────────────────
    if args.from_s3:
        print("\n--- PULL TỪ S3 ---")
        try:
            from app.utils.s3 import get_s3
            from app.config import get_settings
            if not get_settings().S3_ENABLED:
                print("  S3_ENABLED=false trong .env, nhưng --from-s3 được chỉ định.")
            s3 = get_s3()
            conn = s3.check_connection()
            print(f"  {conn['message']}")
            if not conn["ok"]:
                sys.exit(1)
            ok_theory = s3.download_theory(force=True)
            if ok_theory:
                print("  OK: Đã tải theory.json")
        except Exception as e:
            print(f"  Lỗi khi pull từ S3: {e}")
            sys.exit(1)

    kb = get_knowledge_base()

    theory_count  = 0
    formula_count = 0

    if args.formulas:
        formula_count = ingest_formulas(kb)
    elif args.theory:
        theory_count = ingest_theory(kb)
    else:
        theory_count  = ingest_theory(kb)
        formula_count = ingest_formulas(kb)

    print("\n" + "=" * 50)
    print("KẾT QUẢ:")
    stats = kb.get_stats()
    print(f"  math_theory: {stats['theory_docs']} documents (theory + formula)")
    print(f"  Ingested   : {theory_count} theory + {formula_count} formulas")
    print("\nHoàn tất!")


if __name__ == "__main__":
    main()
