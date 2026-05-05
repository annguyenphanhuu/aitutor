"""
Script ingest toàn bộ data vào ChromaDB.

Kiến trúc 2 collection:
  math_theory  ← data/theory.json
  math_exams   ← data/exams/*.json (tất cả file trong thư mục)

Cách dùng:
  python scripts/ingest.py              # ingest tất cả (từ local)
  python scripts/ingest.py --theory     # chỉ ingest lý thuyết
  python scripts/ingest.py --exams      # chỉ ingest đề thi
  python scripts/ingest.py --from-s3    # pull từ S3 rồi ingest tất cả
"""

import json
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag.knowledge_base import get_knowledge_base

DATA_DIR       = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
THEORY_FILE    = os.path.join(DATA_DIR, "theory.json")
EXAMS_DIR      = os.path.join(DATA_DIR, "exams")
VALID_TYPES    = {"theory", "exam_mcq", "exam_true_false", "exam_short_answer"}


def load_json(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def validate_docs(docs: list[dict], source: str) -> list[dict]:
    """Validate và lọc bỏ các doc lỗi, trả về danh sách hợp lệ."""
    valid = []
    for doc in docs:
        doc_id = doc.get("id", "?")
        meta   = doc.get("metadata", {})
        dtype  = meta.get("type", "")

        if not doc.get("id"):
            print(f"  ⚠️  Bỏ qua: thiếu 'id' trong {source}")
            continue
        if not doc.get("content"):
            print(f"  ⚠️  [{doc_id}] Bỏ qua: thiếu 'content'")
            continue
        if dtype not in VALID_TYPES:
            print(f"  ⚠️  [{doc_id}] Bỏ qua: type không hợp lệ '{dtype}'")
            continue
        valid.append(doc)
    return valid


def ingest_theory(kb):
    """Ingest theory.json → collection math_theory."""
    print("\n📘 INGEST LÝ THUYẾT")
    print(f"   Nguồn: {THEORY_FILE}")

    if not os.path.exists(THEORY_FILE):
        print("   ⏭️  File theory.json chưa có, bỏ qua.")
        return 0

    docs  = load_json(THEORY_FILE)
    valid = validate_docs(docs, "theory.json")

    # Chỉ nhận type=theory
    theory_docs = [d for d in valid if d["metadata"]["type"] == "theory"]
    if not theory_docs:
        print("   ❌ Không có document nào hợp lệ.")
        return 0

    kb.ingest_theory(theory_docs)
    print(f"   ✅ Đã ingest {len(theory_docs)} documents vào math_theory")
    return len(theory_docs)


def ingest_exams(kb):
    """Ingest tất cả file JSON trong data/exams/ → collection math_exams."""
    print("\n📝 INGEST ĐỀ THI")
    print(f"   Thư mục: {EXAMS_DIR}")

    if not os.path.exists(EXAMS_DIR):
        print("   ⏭️  Thư mục data/exams/ chưa có, bỏ qua.")
        return 0

    exam_files = sorted([
        f for f in os.listdir(EXAMS_DIR)
        if f.endswith(".json") and not f.startswith("_")
    ])

    if not exam_files:
        print("   ⏭️  Chưa có file đề thi nào trong data/exams/")
        return 0

    total = 0
    all_exam_docs = []
    for filename in exam_files:
        path = os.path.join(EXAMS_DIR, filename)
        docs = load_json(path)
        valid = validate_docs(docs, filename)
        exam_docs = [
            d for d in valid
            if d["metadata"]["type"] in {"exam_mcq", "exam_true_false", "exam_short_answer"}
        ]
        print(f"   📄 {filename:<35} {len(exam_docs)} câu hợp lệ")
        all_exam_docs.extend(exam_docs)
        total += len(exam_docs)

    if all_exam_docs:
        kb.ingest_exams(all_exam_docs)
        print(f"\n   ✅ Đã ingest {total} câu hỏi đề thi vào math_exams")
    else:
        print("   ❌ Không có câu hỏi đề thi nào hợp lệ.")
    return total


def main():
    parser = argparse.ArgumentParser(description="Ingest RAG data vào ChromaDB")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--theory",   action="store_true", help="Chỉ ingest lý thuyết")
    group.add_argument("--exams",    action="store_true", help="Chỉ ingest đề thi")
    group.add_argument("--from-s3",  action="store_true", help="Pull từ S3 rồi ingest tất cả")
    args = parser.parse_args()

    print("🚀 AITutor RAG Ingest")
    print("=" * 50)

    # ── Pull từ S3 trước nếu có flag --from-s3 ────────────────────────
    if args.from_s3:
        print("\n☁️  PULL TỪ S3")
        try:
            from app.utils.s3 import get_s3
            from app.config import get_settings
            if not get_settings().S3_ENABLED:
                print("   ⚠️  S3_ENABLED=false trong .env. Bết cưỡng bằng --from-s3, tiếp tục…")
            s3 = get_s3()
            # Kiểm tra kết nối
            conn = s3.check_connection()
            print(f"   {conn['message']}")
            if not conn["ok"]:
                sys.exit(1)
            # Download
            pulled_exams = s3.download_exams(force=True)
            print(f"   ✔ Được tải {len(pulled_exams)} file đề thi")
            ok_theory = s3.download_theory(force=True)
            if ok_theory:
                print("   ✔ Đã tải theory.json")
        except Exception as e:
            print(f"   ❌ Lỗi khi pull từ S3: {e}")
            sys.exit(1)

    kb = get_knowledge_base()

    theory_count = 0
    exam_count   = 0

    if args.exams:
        exam_count = ingest_exams(kb)
    elif args.theory:
        theory_count = ingest_theory(kb)
    else:
        theory_count = ingest_theory(kb)
        exam_count   = ingest_exams(kb)

    print("\n" + "=" * 50)
    print("📊 KẾT QUẢ:")
    stats = kb.get_stats()
    print(f"   math_theory : {stats['theory_docs']} documents")
    print(f"   math_exams  : {stats['exam_docs']} documents")
    print(f"   Tổng cộng   : {stats['total']} documents")
    print("\n✅ Hoàn tất!")


if __name__ == "__main__":
    main()
