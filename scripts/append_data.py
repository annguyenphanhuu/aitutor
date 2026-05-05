"""
Script APPEND data mới vào đúng file theo type — an toàn, validate trước khi lưu.

Kiến trúc data:
  data/theory.json       ← type: "theory"
  data/exams/*.json      ← type: "exam_mcq" | "exam_true_false" | "exam_short_answer"

Cách dùng:
  1. Dùng LLM sinh JSON array → lưu vào data/new_batch.json
  2. Chạy: python scripts/append_data.py
  3. Script validate → route vào đúng chỗ → in báo cáo
  4. Chạy: python scripts/ingest.py để nạp vào ChromaDB
"""

import json
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.knowledge_tracing.skill_graph import SKILLS

DATA_DIR   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
EXAMS_DIR  = os.path.join(DATA_DIR, "exams")
BATCH_PATH = os.path.join(DATA_DIR, "new_batch.json")

THEORY_FILE = os.path.join(DATA_DIR, "theory.json")

VALID_THEORY_TYPES = {"theory"}
VALID_EXAM_TYPES   = {"exam_mcq", "exam_true_false", "exam_short_answer"}
ALL_VALID_TYPES    = VALID_THEORY_TYPES | VALID_EXAM_TYPES

VALID_SKILLS = set(SKILLS.keys())

# Metadata bắt buộc theo type
REQUIRED_EXAM_META = {"exam_id", "year", "exam_source", "question_number",
                      "difficulty_part", "skill_id", "chapter", "type", "correct_answer"}
REQUIRED_THEORY_META = {"skill_id", "chapter", "type"}


# ── Helpers ────────────────────────────────────────────────

def load_file(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def save_file(path: str, docs: list[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)


# ── Validation ─────────────────────────────────────────────

def validate(docs: list[dict]) -> list[str]:
    errors   = []
    ids_seen = set()

    for i, doc in enumerate(docs):
        label = doc.get("id", f"#index_{i}")

        # Kiểm tra các field bắt buộc
        for field in ("id", "content", "metadata"):
            if field not in doc:
                errors.append(f"[{label}] Thiếu field '{field}'")

        if "metadata" not in doc:
            continue

        meta   = doc.get("metadata", {})
        dtype  = meta.get("type", "")

        # Type hợp lệ?
        if dtype not in ALL_VALID_TYPES:
            errors.append(
                f"[{label}] type không hợp lệ: '{dtype}'\n"
                f"         Chấp nhận: {sorted(ALL_VALID_TYPES)}"
            )
            continue

        # Metadata bắt buộc theo type
        required = REQUIRED_EXAM_META if dtype in VALID_EXAM_TYPES else REQUIRED_THEORY_META
        for req_field in required:
            if req_field not in meta:
                errors.append(f"[{label}] Thiếu metadata['{req_field}'] (bắt buộc cho type '{dtype}')")

        # skill_id hợp lệ?
        skill_id = meta.get("skill_id")
        if skill_id and skill_id not in VALID_SKILLS:
            errors.append(f"[{label}] skill_id không tồn tại: '{skill_id}'")

        # Nội dung đủ dài?
        if len(doc.get("content", "")) < 20:
            errors.append(f"[{label}] content quá ngắn ({len(doc.get('content', ''))} ký tự)")

        # Exam docs: recommend v2 format with separate 'answer' field
        if dtype in VALID_EXAM_TYPES and "answer" not in doc:
            # Warning: old format still accepted but not recommended
            print(f"  ⚠️  [{label}] Khuyến khích dùng field 'answer' riêng (format v2)")

        # ID trùng?
        doc_id = doc.get("id", "")
        if doc_id in ids_seen:
            errors.append(f"[{label}] ID trùng trong batch: '{doc_id}'")
        ids_seen.add(doc_id)

    return errors


# ── Routing ────────────────────────────────────────────────

def route_theory_docs(docs: list[dict]) -> int:
    """Append theory docs vào data/theory.json."""
    theory_docs = [d for d in docs if d["metadata"]["type"] == "theory"]
    if not theory_docs:
        return 0

    existing     = load_file(THEORY_FILE)
    existing_ids = {d["id"] for d in existing}
    added        = [d for d in theory_docs if d["id"] not in existing_ids]
    skipped      = len(theory_docs) - len(added)

    save_file(THEORY_FILE, existing + added)
    msg = f"  📘 theory.json              +{len(added)} docs"
    if skipped:
        msg += f" (bỏ qua {skipped} trùng)"
    print(msg)
    return len(added)


def route_exam_docs(docs: list[dict]) -> int:
    """
    Route từng exam doc vào file tương ứng trong data/exams/{exam_id}.json
    dựa theo metadata['exam_id'].
    """
    exam_docs = [d for d in docs if d["metadata"]["type"] in VALID_EXAM_TYPES]
    if not exam_docs:
        return 0

    os.makedirs(EXAMS_DIR, exist_ok=True)

    # Group theo exam_id
    by_exam: dict[str, list] = {}
    for doc in exam_docs:
        eid = doc["metadata"].get("exam_id", "unknown")
        by_exam.setdefault(eid, []).append(doc)

    total_added = 0
    for exam_id, edocs in sorted(by_exam.items()):
        filename = f"{exam_id}.json"
        path     = os.path.join(EXAMS_DIR, filename)

        existing     = load_file(path)
        existing_ids = {d["id"] for d in existing}
        added        = [d for d in edocs if d["id"] not in existing_ids]
        skipped      = len(edocs) - len(added)

        save_file(path, existing + added)
        msg = f"  📝 exams/{filename:<27} +{len(added)} câu"
        if skipped:
            msg += f" (bỏ qua {skipped} trùng)"
        print(msg)
        total_added += len(added)

    return total_added


# ── Coverage Report ────────────────────────────────────────

def coverage_summary():
    """In bảng coverage: mỗi skill có bao nhiêu lý thuyết + câu thi."""
    theory_docs = load_file(THEORY_FILE)

    exam_docs = []
    if os.path.exists(EXAMS_DIR):
        for f in os.listdir(EXAMS_DIR):
            if f.endswith(".json"):
                exam_docs.extend(load_file(os.path.join(EXAMS_DIR, f)))

    # Group by skill_id
    coverage: dict = {}
    for doc in theory_docs + exam_docs:
        meta  = doc.get("metadata", {})
        sid   = meta.get("skill_id", "?")
        dtype = meta.get("type", "?")
        coverage.setdefault(sid, {"theory": 0, "exam": 0})
        if dtype == "theory":
            coverage[sid]["theory"] += 1
        elif dtype in VALID_EXAM_TYPES:
            coverage[sid]["exam"] += 1

    print("\n📊 COVERAGE")
    print(f"  {'Skill':<38} {'Theory':>8} {'Exam':>8} {'Total':>8}")
    print("  " + "─" * 62)

    grand_theory = grand_exam = 0
    for skill_id, info in SKILLS.items():
        name   = info["name"][:36]
        counts = coverage.get(skill_id, {})
        t      = counts.get("theory", 0)
        e      = counts.get("exam",   0)
        grand_theory += t
        grand_exam   += e
        t_cell = "❌" if t == 0 else str(t)
        e_cell = "❌" if e == 0 else str(e)
        print(f"  {name:<38} {t_cell:>8} {e_cell:>8} {t+e:>8}")

    print("  " + "─" * 62)
    print(f"  {'TỔNG':<38} {grand_theory:>8} {grand_exam:>8} {grand_theory+grand_exam:>8}")


# ── Main ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch", default=BATCH_PATH,
        help="Đường dẫn tới file JSON batch (mặc định: data/new_batch.json)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.batch):
        print(f"❌ Không tìm thấy: {args.batch}")
        print("   Lưu JSON từ LLM vào file đó rồi chạy lại.")
        sys.exit(1)

    with open(args.batch, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    try:
        new_docs = json.loads(raw)
        if isinstance(new_docs, dict):
            new_docs = [new_docs]
    except json.JSONDecodeError as e:
        print(f"❌ JSON không hợp lệ:\n   {e}")
        sys.exit(1)

    print(f"📥 Đọc {len(new_docs)} documents từ {os.path.basename(args.batch)}")

    # Validate
    errors = validate(new_docs)
    if errors:
        print(f"\n⚠️  {len(errors)} lỗi validation:")
        for e in errors:
            print(f"   • {e}")
        print("\n❌ Dừng lại. Hãy sửa lỗi trước.")
        sys.exit(1)

    print("✅ Validation passed!\n")

    # Route
    theory_added = route_theory_docs(new_docs)
    exam_added   = route_exam_docs(new_docs)

    print(f"\n✅ Tổng thêm: {theory_added + exam_added} "
          f"(theory: {theory_added}, exam: {exam_added})")

    # Coverage report
    coverage_summary()

    # Xóa batch sau khi xử lý
    if args.batch == BATCH_PATH:
        os.remove(BATCH_PATH)
        print(f"\n🗑️  Đã xóa {os.path.basename(BATCH_PATH)}")

    print("👉 Bước tiếp: python scripts/ingest.py")


if __name__ == "__main__":
    main()
