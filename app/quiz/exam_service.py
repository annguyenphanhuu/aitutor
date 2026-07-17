"""Service for loading and grading pre-built exam sets from data/exams/.

S3 Integration:
  Nếu S3_ENABLED=true trong .env, service sẽ:
  - list_exams(): sync mọi file đề thi còn thiếu từ S3 trước khi list local
  - load_exam(): fallback tải từ S3 nếu file không tồn tại local
  - grade_exam(): cùng fallback với load_exam()
"""

import json
import os
import re
import glob
import logging
from typing import Any

from app.quiz.grading import score_short_answer, score_true_false

log = logging.getLogger(__name__)

EXAMS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "exams")

# exam_id comes from the URL path — restrict to safe filename characters
# to prevent path traversal (e.g. "../../secrets").
_EXAM_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


def _is_valid_exam_id(exam_id: str) -> bool:
    return bool(exam_id and _EXAM_ID_RE.match(exam_id))


def _sync_from_s3_if_enabled(force: bool = False) -> None:
    """Nếu S3_ENABLED, kéo các file đề thi còn thiếu từ S3 về local (non-blocking)."""
    try:
        from app.config import get_settings
        if not get_settings().S3_ENABLED:
            return
        from app.utils.s3 import get_s3
        s3  = get_s3()
        pulled = s3.download_exams(force=force)
        if pulled:
            log.info(f"[S3] Đã tải {len(pulled)} file đề thi từ S3: {pulled}")
    except Exception as e:
        log.warning(f"[S3] Không thể sync exams từ S3 (ứng dụng vẫn chạy bình thường): {e}")


def _download_single_exam_from_s3(exam_id: str) -> bool:
    """Thử tải một file đề thi cụ thể từ S3. Returns True nếu thành công."""
    try:
        from app.config import get_settings
        if not get_settings().S3_ENABLED:
            return False
        from app.utils.s3 import get_s3
        get_s3().download_exam(f"{exam_id}.json")
        log.info(f"[S3] Đã tải {exam_id}.json từ S3")
        return True
    except Exception as e:
        log.warning(f"[S3] Không thể tải {exam_id}.json từ S3: {e}")
        return False


# ── List available exams ────────────────────────────────────────────────────

def list_exams() -> list[dict]:
    """Scan data/exams/*.json and return exam metadata (sync từ S3 nếu enabled)."""
    # Kéo file còn thiếu từ S3 trước khi scan local
    _sync_from_s3_if_enabled(force=False)

    exams = []
    pattern = os.path.join(EXAMS_DIR, "*.json")
    for filepath in sorted(glob.glob(pattern)):
        basename = os.path.basename(filepath)
        if basename.startswith("_"):
            continue
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                questions = json.load(f)
            if not questions:
                continue
            meta = questions[0].get("metadata", {})

            # Count by type
            type_counts = {}
            for q in questions:
                qtype = q.get("metadata", {}).get("type", "unknown")
                type_counts[qtype] = type_counts.get(qtype, 0) + 1

            exams.append({
                "exam_id": meta.get("exam_id", basename.replace(".json", "")),
                "exam_source": meta.get("exam_source", ""),
                "year": meta.get("year", 0),
                "question_count": len(questions),
                "type_counts": type_counts,
                "filename": basename,
            })
        except Exception as e:
            log.warning(f"Skipping {basename}: {e}")
    return exams


# ── Load exam questions (hide answers for test-taking) ──────────────────────

def _parse_question_for_student(q: dict) -> dict:
    """Parse a raw exam question into student-facing format (no answer/explanation)."""
    meta = q.get("metadata", {})
    qtype = meta.get("type", "exam_mcq")
    content = q.get("content", "")

    # New format: content is already clean (no answer).
    # Backward compat: if content still has ĐÁP ÁN:, strip it.
    question_text = content
    if "ĐÁP ÁN:" in content:
        question_text = content.split("ĐÁP ÁN:", 1)[0].strip()

    # Strip explicit [Xem hình: ...] markers since frontend renders images separately
    # BUT for true_false questions, keep them so _parse_tf_content can assign images to statements
    if qtype != "exam_true_false":
        question_text = re.sub(r'\[Xem hình:\s*([^\]]+)\]', '', question_text).strip()

    # Build image path lookup: filename -> full_path
    image_path_str = meta.get("image_path", "")
    image_lookup = {}
    if image_path_str:
        for p in image_path_str.split(","):
            p = p.strip()
            if p:
                fname = p.rsplit("/", 1)[-1] if "/" in p else p
                image_lookup[fname] = p

    qtype_map = {
        "exam_mcq": "mcq",
        "exam_true_false": "true_false",
        "exam_short_answer": "short_answer"
    }
    result: dict[str, Any] = {
        "id": q.get("id", ""),
        "question_number": meta.get("question_number", 0),
        "type": qtype,
        "question_type": qtype_map.get(qtype, "mcq"),
        "difficulty_part": meta.get("difficulty_part", 1),
        "chapter": meta.get("chapter", ""),
        "skill_id": meta.get("skill_id", ""),
        "skill_ids": meta.get("skill_ids") or ([meta.get("skill_id")] if meta.get("skill_id") else []),
        "formula_ids": meta.get("formula_ids") or [],
        "has_image": meta.get("has_image", False),
        "image_path": image_path_str,
    }

    if qtype == "exam_mcq":
        # Parse question text and choices
        q_body, choices = _parse_mcq_content(question_text)
        result["question_text"] = q_body
        result["choices"] = choices

    elif qtype == "exam_true_false":
        # Parse question text and statements with image references
        q_body, statements, stmt_images = _parse_tf_content(question_text, image_lookup)
        result["question_text"] = q_body
        result["statements"] = statements
        result["statement_images"] = stmt_images
        # Determine which images are statement-specific vs question-level
        claimed = set(stmt_images.values())
        unclaimed = [p for p in image_lookup.values() if p not in claimed]
        result["question_images"] = unclaimed  # images not tied to any statement

    elif qtype == "exam_short_answer":
        # Just the question text
        q_body = question_text
        # Remove trailing choices/options if any
        result["question_text"] = q_body

    else:
        result["question_text"] = question_text

    return result


def _parse_mcq_content(text: str) -> tuple[str, list[str]]:
    """Extract question body and 4 choices from MCQ content."""
    # Split at answer choices A. B. C. D.
    pattern = r'\n\s*([A-D])\.\s+'
    parts = re.split(pattern, text)

    if len(parts) >= 9:  # body + 4*(label+text)
        body = parts[0].strip()
        choices = []
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                choices.append(parts[i + 1].strip())
        return body, choices[:4]

    # Fallback: try to find A. B. C. D. lines
    lines = text.strip().split('\n')
    body_lines = []
    choices = []
    for line in lines:
        stripped = line.strip()
        match = re.match(r'^([A-D])\.\s+(.+)', stripped)
        if match:
            choices.append(match.group(2).strip())
        else:
            if not choices:  # Still in body
                body_lines.append(stripped)

    return '\n'.join(body_lines), choices[:4]


def _strip_table_suffix(text: str) -> str:
    """
    Strip trailing Markdown table separators (| col | col) from a statement text,
    but ONLY for pipe characters that appear OUTSIDE LaTeX math mode ($...$).
    Finds the FIRST pipe outside math and strips from there.
    For example:
      '$P(Y |X) = 0,07$'        -> kept as-is (| is inside math)
      '$P(Y |X)$ | Đúng | Sai' -> stripped to '$P(Y |X)$'
      '$f(x)$ | Đúng | Sai |'  -> stripped to '$f(x)$'
    """
    in_math = False
    for i, ch in enumerate(text):
        if ch == '$':
            in_math = not in_math
        elif ch == '|' and not in_math:
            # First pipe outside LaTeX math — strip from here onward
            return text[:i].rstrip()
    return text


def _parse_tf_content(text: str, image_lookup: dict | None = None) -> tuple[str, list[dict], dict]:
    """
    Extract question body and statements from True/False content.
    Also detects [Xem hình: filename] refs in statements and maps them to full paths.
    Returns: (body, statements, statement_images)  where statement_images = {"b": "data/...", "d": "data/..."}
    """
    if image_lookup is None:
        image_lookup = {}

    lines = text.strip().split('\n')
    body_lines = []
    statements = []
    statement_images = {}  # label -> image_path
    in_statements = False
    current_label = None

    for line in lines:
        stripped = line.strip()
        stripped_table = stripped.lstrip('|').strip()
        
        # Check for statement pattern: a) ... or (a) ...
        # Can be followed by optional spaces and optional pipes like: (a)| text or (a)  | text
        stmt_match = re.match(r'^([a-d])\)[\s\|]*(.+)', stripped_table)
        if not stmt_match:
            stmt_match = re.match(r'^\(([a-d])\)[\s\|]*(.+)', stripped_table)

        if stmt_match:
            in_statements = True
            current_label = stmt_match.group(1)
            stmt_text = stmt_match.group(2).strip()
            stmt_text = _strip_table_suffix(stmt_text)
            
            # Remove → ĐÚNG/SAI markers
            stmt_text = re.sub(r'\s*→\s*(ĐÚNG|SAI|Đúng|Sai)\s*$', '', stmt_text)
            # Extract [Xem hình: filename] references
            img_ref = re.search(r'\[Xem hình:\s*([^\]]+)\]', stmt_text)
            if img_ref:
                img_fname = img_ref.group(1).strip()
                full_path = image_lookup.get(img_fname, img_fname)
                statement_images[current_label] = full_path
                # Remove the [Xem hình: ...] from display text
                stmt_text = re.sub(r'\s*\[Xem hình:\s*[^\]]+\]\s*', ' ', stmt_text).strip()
            statements.append({"text": stmt_text})
        elif in_statements and current_label:
            # Continuation line: might contain [Xem hình: filename] or table data for current statement
            img_ref = re.search(r'\[Xem hình:\s*([^\]]+)\]', stripped)
            if img_ref:
                img_fname = img_ref.group(1).strip()
                full_path = image_lookup.get(img_fname, img_fname)
                statement_images[current_label] = full_path
            elif stripped and not stripped.startswith('|') and not stripped.startswith('---'):
                # Append continuation text to current statement
                if statements:
                    cleaned = re.sub(r'\s*\[Xem hình:\s*[^\]]+\]\s*', ' ', stripped).strip()
                    if cleaned:
                        statements[-1]["text"] += '\n' + cleaned
        elif not in_statements:
            if stripped.startswith('| Mệnh đề') or stripped.startswith('|---') or stripped.startswith('| Mệnh đề | Đúng | Sai |'):
                continue
            body_lines.append(stripped)

    # If no explicit statements found, try table format
    if not statements:
        return text.strip(), [], {}

    return '\n'.join(body_lines), statements, statement_images


def load_exam(exam_id: str) -> dict | None:
    """Load all questions for an exam, formatted for student test-taking.

    Fallback: nếu file không có local, thử tải từ S3 (khi S3_ENABLED=true).
    """
    if not _is_valid_exam_id(exam_id):
        return None
    filepath = os.path.join(EXAMS_DIR, f"{exam_id}.json")
    if not os.path.exists(filepath):
        # Thử pull từ S3 trước khi bỏ cuộc
        if not _download_single_exam_from_s3(exam_id):
            return None
        # Kiểm tra lại sau khi download
        if not os.path.exists(filepath):
            return None

    with open(filepath, "r", encoding="utf-8") as f:
        raw_questions = json.load(f)

    meta = raw_questions[0].get("metadata", {}) if raw_questions else {}

    questions = [_parse_question_for_student(q) for q in raw_questions]

    return {
        "exam_id": exam_id,
        "exam_source": meta.get("exam_source", ""),
        "year": meta.get("year", 0),
        "question_count": len(questions),
        "questions": questions,
    }


# ── Grade exam ──────────────────────────────────────────────────────────────

def _grade_mcq(correct: str, answer: str | None) -> tuple[float, float, bool]:
    """Grade MCQ: 0.25 per correct. Returns (earned, max, is_correct)."""
    max_pts = 0.25
    if not answer:
        return 0.0, max_pts, False
    is_correct = answer.upper().strip() == correct.upper().strip()
    return (max_pts if is_correct else 0.0), max_pts, is_correct


def _grade_true_false(correct_str: str, answers: dict | None) -> tuple[float, float, list[dict]]:
    """
    Grade True/False question.
    correct_str: "a-T,b-F,c-T,d-F"
    answers: {"a": true, "b": false, "c": true, "d": true}
    Scoring: 1 correct=0.1, 2=0.25, 3=0.5, 4=1.0
    Returns (earned, max, detail_list)
    """
    max_pts = 1.0
    # Parse correct answers
    correct_map = {}
    for part in correct_str.split(","):
        part = part.strip()
        match = re.match(r'([a-d])\s*-\s*([TFĐStfđs])', part)
        if match:
            label = match.group(1).lower()
            val = match.group(2).upper() in ('T', 'Đ')
            correct_map[label] = val

    if not answers:
        answers = {}

    details = []
    correct_count = 0
    for label in ['a', 'b', 'c', 'd']:
        if label not in correct_map:
            continue
        correct_val = correct_map[label]
        student_val = answers.get(label)
        is_right = student_val is not None and student_val == correct_val
        if is_right:
            correct_count += 1
        details.append({
            "label": label,
            "correct": correct_val,
            "student": student_val,
            "is_correct": is_right,
        })

    earned, _ = score_true_false(
        [answers.get(label) for label in correct_map],
        list(correct_map.values()),
    )
    return earned, max_pts, details


def _grade_short_answer(correct: str, answer: str | None) -> tuple[float, float, bool]:
    """Grade short answer: 0.5 per correct. Normalize numbers for comparison."""
    max_pts = 0.5
    if not answer:
        return 0.0, max_pts, False

    earned, is_correct = score_short_answer(
        answer,
        correct,
        max_points=max_pts,
        numeric_tolerance=0.0,
        remove_spaces=True,
    )
    return earned, max_pts, is_correct


def grade_exam(exam_id: str, answers: dict) -> dict | None:
    """
    Grade an entire exam.
    answers format: {
        "question_id": {
            "mcq_answer": "B",           # for MCQ
            "tf_answers": {"a": true...}, # for True/False
            "sa_answer": "42"             # for Short Answer
        }
    }
    Fallback: nếu file không có local, thử tải từ S3 (khi S3_ENABLED=true).
    """
    if not _is_valid_exam_id(exam_id):
        return None
    filepath = os.path.join(EXAMS_DIR, f"{exam_id}.json")
    if not os.path.exists(filepath):
        if not _download_single_exam_from_s3(exam_id):
            return None
        if not os.path.exists(filepath):
            return None

    with open(filepath, "r", encoding="utf-8") as f:
        raw_questions = json.load(f)

    total_earned = 0.0
    total_max = 0.0
    results = []

    for q in raw_questions:
        qid = q.get("id", "")
        meta = q.get("metadata", {})
        qtype = meta.get("type", "exam_mcq")
        correct = meta.get("correct_answer", "")
        answer_field = q.get("answer", "")

        student_answer = answers.get(qid, {})

        # Extract explanation from dedicated 'answer' field
        explanation = ""
        if answer_field:
            if "GIẢI THÍCH:" in answer_field:
                explanation = answer_field.split("GIẢI THÍCH:", 1)[1].strip()
            else:
                explanation = answer_field.strip()
        else:
            # Backward compat: try old format (content contains everything)
            content = q.get("content", "")
            if "GIẢI THÍCH:" in content:
                explanation = content.split("GIẢI THÍCH:", 1)[1].strip()
            elif "ĐÁP ÁN:" in content:
                explanation = content.split("ĐÁP ÁN:", 1)[1].strip()

        # Parse inline images in explanation
        if explanation:
            def replace_inline_image(match):
                img_name = match.group(1).strip()
                # Determine the exam_id prefix for the folder path
                safe_exam_id = exam_id.strip()
                return f'<br><img src="/data/exams/images/{safe_exam_id}/{img_name}" class="quiz-explanation-img" style="max-width:100%; border-radius:8px; margin-top:10px;"><br>'
            explanation = re.sub(r'\[Xem hình:\s*([^\]]+)\]', replace_inline_image, explanation)

        result_item: dict[str, Any] = {
            "id": qid,
            "question_number": meta.get("question_number", 0),
            "type": qtype,
            "difficulty_part": meta.get("difficulty_part", 1),
            "chapter": meta.get("chapter", ""),
            "skill_id": meta.get("skill_id", ""),
            "skill_ids": meta.get("skill_ids") or ([meta.get("skill_id")] if meta.get("skill_id") else []),
            "formula_ids": meta.get("formula_ids") or [],
            "correct_answer": correct,
            "explanation": explanation,
        }

        if qtype == "exam_mcq":
            earned, max_pts, is_correct = _grade_mcq(correct, student_answer.get("mcq_answer"))
            result_item["student_answer"] = student_answer.get("mcq_answer", "")
            result_item["is_correct"] = is_correct
            result_item["points_earned"] = earned
            result_item["points_max"] = max_pts

        elif qtype == "exam_true_false":
            earned, max_pts, details = _grade_true_false(correct, student_answer.get("tf_answers"))
            result_item["student_answer"] = student_answer.get("tf_answers", {})
            result_item["statement_details"] = details
            correct_count = sum(1 for d in details if d["is_correct"])
            result_item["is_correct"] = correct_count == len(details)
            result_item["correct_statements"] = correct_count
            result_item["total_statements"] = len(details)
            result_item["points_earned"] = earned
            result_item["points_max"] = max_pts

        elif qtype == "exam_short_answer":
            earned, max_pts, is_correct = _grade_short_answer(correct, student_answer.get("sa_answer"))
            result_item["student_answer"] = student_answer.get("sa_answer", "")
            result_item["is_correct"] = is_correct
            result_item["points_earned"] = earned
            result_item["points_max"] = max_pts

        else:
            earned, max_pts = 0.0, 0.0

        total_earned += earned
        total_max += max_pts
        results.append(result_item)

    # Part scores
    part_scores = {}
    for r in results:
        part = r.get("difficulty_part", 1)
        part_label = {1: "mcq", 2: "true_false", 3: "short_answer"}.get(part, f"part_{part}")
        part_scores[part_label] = part_scores.get(part_label, 0.0) + r.get("points_earned", 0.0)

    score_pct = (total_earned / total_max * 100) if total_max > 0 else 0

    return {
        "exam_id": exam_id,
        "total_score": round(total_earned, 2),
        "max_score": round(total_max, 2),
        "score_percent": round(score_pct, 1),
        "part_scores": {k: round(v, 2) for k, v in part_scores.items()},
        "total_questions": len(results),
        "results": results,
    }


# ── Grade Exam + Update BKT Mastery (async orchestration) ──────────────────

async def grade_exam_and_update(
    db,
    exam_id: str,
    answers: dict,
    user_id: int,
) -> dict | None:
    """
    Grade an exam AND persist BKT mastery updates, InteractionLog entries,
    and Spaced Repetition cards for all skills encountered.

    Returns the standard graded result dict (same as grade_exam()),
    enriched with a 'mastery_updates' field: { skill_id: new_p_mastery }.
    """
    from app.knowledge_tracing.service import batch_update_mastery_from_exam
    from app.spaced_repetition.service import ensure_cards_for_attempted_skills
    from app.db.models import InteractionLog

    # Step 1: Grade the exam (sync — no DB writes)
    result = grade_exam(exam_id, answers)
    if result is None:
        return None

    # Step 2: Extract (skill_id, is_correct) pairs from graded results
    exam_items = []
    for r in result["results"]:
        sid = r.get("skill_id") or r.get("chapter")
        if sid:
            exam_items.append({
                "skill_id": sid,
                "is_correct": r.get("is_correct", False),
            })

    # Step 3: Batch BKT update (sequential per skill to preserve chain)
    mastery_updates: dict[str, float] = {}
    if exam_items:
        mastery_updates = await batch_update_mastery_from_exam(db, exam_items, user_id)

    # Step 4: Log each question into InteractionLog (feeds Dashboard recent_interactions)
    for r in result["results"]:
        sid = r.get("skill_id") or r.get("chapter")
        log = InteractionLog(
            user_id=user_id,
            question=f"[Đề {exam_id}] Câu {r.get('question_number', '')}",
            agent_response=None,
            skill_id=sid or None,
            is_correct=r.get("is_correct"),
            response_mode="exam",
        )
        db.add(log)

    # Step 5: Create/ensure SR cards for all unique skills in the exam
    seen_skills = {r.get("skill_id") for r in result["results"] if r.get("skill_id")}
    for sid in seen_skills:
        await ensure_cards_for_attempted_skills(db, sid, user_id)

    await db.flush()

    result["mastery_updates"] = mastery_updates
    return result
