"""
AITutor — Custom Evaluation Monitor

Chọn đề và câu cụ thể để evaluate, hiển thị toàn bộ trace:
  - System prompt gửi đến LLM
  - RAG chunks được retrieve (với score)
  - Response của model
  - MathJudge score
  - RAGAS score (tùy chọn)

Chạy:
    # Xem tất cả câu đề 01
    python scripts/run_eval_monitor.py --exam de01

    # Chọn câu cụ thể
    python scripts/run_eval_monitor.py --exam de01 --questions 1,2,19

    # Chọn range
    python scripts/run_eval_monitor.py --exam de01 --questions 1-5

    # Chỉ MCQ, không RAGAS (nhanh hơn)
    python scripts/run_eval_monitor.py --exam de01 --questions 1,2 --no-ragas

    # Chọn đề khác
    python scripts/run_eval_monitor.py --exam de05 --questions 3,7 --types exam_short_answer

    # Lưu trace ra file
    python scripts/run_eval_monitor.py --exam de01 --questions 1,2 --save-trace
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import argparse
import sys
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.WARNING,   # Tắt INFO noise; script tự in trace
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
EXAMS_DIR    = PROJECT_ROOT / "data" / "exams"
EVAL_DIR     = PROJECT_ROOT / "data" / "evaluation"

EXAM_FILES = {
    "de01": EXAMS_DIR / "de7plus_de01.json",
    "de03": EXAMS_DIR / "de7plus_de03.json",
    "de04": EXAMS_DIR / "de7plus_de04.json",
    "de05": EXAMS_DIR / "de7plus_de05.json",
}

SEP  = "─" * 80
SEP2 = "═" * 80


# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_question_selection(spec: str) -> set[int]:
    """Parse '1,3,5-8,12' → {1, 3, 5, 6, 7, 8, 12}"""
    nums: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            nums.update(range(int(a), int(b) + 1))
        else:
            nums.add(int(part))
    return nums


def clean_question_for_eval(raw: str) -> str:
    """
    Loại bỏ rác khỏi MCQ question trước khi gửi cho evaluators.
    Mục tiêu: giữ lại câu hỏi thuần, bỏ header đề thi + 4 đáp án A/B/C/D.
    Cải thiện Answer Relevancy và Examiner accuracy.
    """
    # Xóa dòng header kiểu "ĐỀ 7+ (ĐỀ số 1) - CÂU 1 [Phần 1 - Trắc nghiệm]:"
    text = re.sub(r"^ĐỀ.*?\n+", "", raw.strip(), flags=re.MULTILINE)
    # Xóa các dòng đáp án MCQ (dòng bắt đầu bằng A. / B. / C. / D.)
    text = re.sub(r"^[ABCD][\.\)].+$", "", text, flags=re.MULTILINE)
    # Xóa nhiều dòng trống liên tiếp
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def _resolve_image(image_path: str) -> Optional[tuple[str, str]]:
    """Trả về (base64_str, media_type) hoặc None."""
    if not image_path:
        return None
    # Lấy ảnh đầu tiên nếu có nhiều (comma-separated)
    first = image_path.split(",")[0].strip()
    p = PROJECT_ROOT / first
    if not p.exists():
        return None
    try:
        b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
        mt  = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(
            p.suffix.lower().lstrip("."), "image/png"
        )
        return b64, mt
    except Exception:
        return None


# ── Load questions ─────────────────────────────────────────────────────────────

def load_questions(
    exam: str,
    question_nums: Optional[set[int]],
    types: Optional[list[str]],
) -> list[dict]:
    exam_file = EXAM_FILES.get(exam)
    if not exam_file or not exam_file.exists():
        print(f"❌ Không tìm thấy file đề: {exam}")
        sys.exit(1)

    items = json.loads(exam_file.read_text(encoding="utf-8"))
    allowed_types = types or ["exam_mcq", "exam_true_false", "exam_short_answer"]

    questions = []
    for item in items:
        meta = item.get("metadata", {})
        if meta.get("type") not in allowed_types:
            continue
        qnum = meta.get("question_number")
        if question_nums and qnum not in question_nums:
            continue
        questions.append({
            "id":             item["id"],
            "question":       item.get("content", ""),
            "ground_truth":   item.get("answer", ""),
            "skill_id":       meta.get("skill_id", ""),
            "chapter":        meta.get("chapter", ""),
            "type":           meta.get("type", ""),
            "correct_answer": meta.get("correct_answer", ""),
            "has_image":      meta.get("has_image", False),
            "image_path":     meta.get("image_path", ""),
            "question_number": qnum,
        })

    questions.sort(key=lambda q: q.get("question_number") or 0)
    return questions


# ── Retrieve RAG ───────────────────────────────────────────────────────────────

def retrieve_with_trace(
    question: str,
    skill_id: str,
    chapter: str,
    k: int = 5,
) -> tuple[list[dict], "QueryExpansion"]:
    """
    Retrieve + trả về (chunks, expansion).

    Thay đổi so với phiên bản cũ:
    - Dùng QueryExpander (LLM) để suy luận chapter/skill/formula từ nội dung câu hỏi
      thay vì dùng trực tiếp metadata đề thi (vốn không đầy đủ và không nhất quán)
    - Soft boost giảm xuống nhẹ hơn (0.06/0.04) để không overwrite semantic score
    - Metadata đề thi (skill_id, chapter) chỉ còn dùng cho evaluation, không dùng để RAG
    """
    from app.rag.knowledge_base import get_knowledge_base
    from app.rag.reranker import get_reranker
    from app.rag.query_expander import expand_question, QueryExpansion

    kb = get_knowledge_base()
    reranker = get_reranker()

    from app.config import get_settings
    settings = get_settings()
    pool_k = settings.RERANKER_CANDIDATE_K if settings.RERANKER_ENABLED else k * 2

    # ── Step 1: Query Expansion — LLM suy luận chapter/skill/formula ──────────
    expansion: QueryExpansion = expand_question(question)
    if expansion.error:
        logger.warning("QueryExpander failed: %s — falling back to raw query", expansion.error)

    # ── Step 2: Candidate retrieval ────────────────────────────────────────────
    candidates = kb.search_theory(question, k=pool_k, alpha=0.55)

    # ── Step 3: Soft boost (nhẹ hơn) dựa trên QueryExpansion, không dùng metadata đề thi ──
    # Boost nhỏ (0.06/0.04) chỉ để điều chỉnh nhẹ ranking, không overwrite semantic score
    SKILL_BOOST   = 0.06
    CHAPTER_BOOST = 0.04

    expanded_skills   = {s.lower() for s in expansion.skill_ids}
    expanded_chapters = {c.lower() for c in expansion.chapters}

    for r in candidates:
        bonus = 0.0
        meta  = r.get("metadata", {})
        chunk_skill   = str(meta.get("skill_id", "")).lower().strip()
        chunk_chapter = str(meta.get("chapter",  "")).lower().strip()

        # Khớp với kết quả QueryExpander (không phải metadata đề thi)
        if chunk_skill and chunk_skill in expanded_skills:
            bonus += SKILL_BOOST
        if chunk_chapter and any(
            chunk_chapter in exp_ch or exp_ch in chunk_chapter
            for exp_ch in expanded_chapters
        ):
            bonus += CHAPTER_BOOST

        if bonus > 0:
            r["hybrid_score"] = min(r.get("hybrid_score", 0) + bonus, 1.0)
            r["_boosted"] = True

    ranked = reranker.rerank(question, candidates, k=k)
    return ranked, expansion


# ── Build prompt (mirrors dataset_builder.py) ──────────────────────────────────

def build_system_prompt(context: str, question_type: str) -> str:
    if question_type == "exam_true_false":
        fmt = (
            "\n\nQUAN TRONG: Cau hoi nay la dang Dung/Sai nhieu menh de. "
            "Bat buoc ket thuc cau tra loi bang dong: "
            "'KET QUA: a-T,b-F,c-T,d-F' (T=Dung, F=Sai)."
        )
    elif question_type == "exam_mcq":
        fmt = (
            "\n\nQUAN TRONG: Bat dau cau tra loi bang 'Chon X.' "
            "voi X la dap an dung (A, B, C hoac D), sau do giai thich ngan gon."
        )
    elif question_type == "exam_short_answer":
        fmt = "\n\nQUAN TRONG: Ket thuc cau tra loi bang 'DAP AN: [gia tri so]'."
    else:
        fmt = ""

    cot = (
        "\n\nQUY TRINH SUY LUAN (BAT BUOC) — Chain of Thought:"
        "\nTruoc khi dua ra dap an, ban PHAI suy luan tung buoc:"
        "\n  Buoc 1: Xac dinh dang bai va phuong phap giai."
        "\n  Buoc 2: Neu bai co HINH VE hoac DO THI, bat buoc thuc hien day du 3 micro-buoc sau:"
        "\n    [Mo ta] - Liet ke tat ca cac gia tri tren truc Ox va Oy bao gom ca dau am (-)."
        "\n             - Xac dinh so nhanh do thi va chung nam o goc phan tu nao."
        "\n             - Mo ta cac duong tiem can (ngang/dung) va cac diem dac biet ro rang tren do thi."
        "\n    [Dinh vi] - Voi moi duong tiem can ngang: xac dinh TRUOC no nam TREN hay DUOI truc Ox,"
        "\n               sau do dem o vuong luoi (grid) de doc chinh xac gia tri y (bao gom dau am)."
        "\n             - Voi moi duong tiem can dung: xac dinh no nam BEN TRAI hay BEN PHAI truc Oy,"
        "\n               dem o vuong luoi de doc chinh xac gia tri x."
        "\n    [Xac nhan] - The toa do vua doc vao do thi de kiem tra tinh hop le truoc khi ket luan."
        "\n  Buoc 3: Thuc hien phep tinh / loai tru phuong an."
        "\n  Buoc 4: Kiem tra lai dap an bang thu nguoc hoac dieu kien bien."
    )

    rag = (
        "\n\nCACH SU DUNG TAI LIEU THAM KHAO:"
        "\nPhan tai lieu ben duoi duoc truy xuat TU DONG tu co so du lieu (RAG)."
        "\nBan PHAI:"
        "\n  - Danh gia xem tai lieu co thuc su lien quan den cau hoi khong truoc khi dung."
        "\n  - Neu tai lieu lien quan -> tan dung de ho tro lap luan va giai thich."
        "\n  - Neu tai lieu khong lien quan -> bo qua va dung kien thuc Toan noi tai de tra loi."
        "\n  - Luon uu tien suy luan Toan hoc dung, tai lieu chi la bo tro."
    )

    return (
        "Ban la gia su Toan 12. Hay tra loi cau hoi ngan gon va chinh xac."
        + fmt + cot + rag
        + "\n\nTAI LIEU THAM KHAO (noi dung chinh xac, muc do lien quan can danh gia):\n"
        + context
    )


# ── Generate answer ────────────────────────────────────────────────────────────

async def generate_with_trace(
    question: str,
    system_prompt: str,
    question_type: str,
    image_path: str,
) -> tuple[str, str, bool]:
    """Returns (response, model_used, used_vision)."""
    from langchain_openai import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage
    from app.config import get_settings

    settings = get_settings()

    img_data = _resolve_image(image_path) if image_path else None
    using_vision = img_data is not None

    model = settings.VISION_LLM_MODEL if using_vision else settings.LLM_MODEL

    llm = ChatOpenAI(
        model=model,
        api_key=settings.OPENAI_API_KEY,
        temperature=0.1,
    )

    if using_vision:
        b64, mt = img_data
        human_content = [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {
                "url": f"data:{mt};base64,{b64}", "detail": "high"
            }},
        ]
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_content)]
    else:
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=question)]

    resp = await llm.ainvoke(messages)
    return resp.content, model, using_vision


# ── MathJudge (inline, no asyncio.run conflict) ────────────────────────────────

async def judge_sample_async(sample: dict, judge_model: str) -> dict:
    """
    Full evaluation stack:
      - MathJudge accuracy (deterministic)
      - LLMExaminer: rubric 4 chiều (accuracy/method/steps/knowledge_use)
      - Visual reasoning judge (khi has_image=True)
    """
    from app.evaluation.math_judge import (
        judge_mcq_accuracy,
        judge_true_false_accuracy,
        judge_short_answer_accuracy,
        judge_examiner,
        judge_visual_reasoning,
    )

    meta      = sample.get("_meta", {})
    qtype     = meta.get("type", "")
    correct   = meta.get("correct_answer", "")
    has_image = meta.get("has_image", False)
    resp      = sample.get("response", "")
    q_raw     = sample.get("user_input", "")
    ref       = sample.get("reference", "")

    # Deterministic accuracy
    if qtype == "exam_mcq":
        acc = judge_mcq_accuracy(resp, correct)
    elif qtype == "exam_true_false":
        acc = judge_true_false_accuracy(resp, correct)
    elif qtype == "exam_short_answer":
        acc = judge_short_answer_accuracy(resp, correct)
    else:
        acc = 0.0

    # LLMExaminer (4-criterion rubric)
    examiner = await judge_examiner(
        question=q_raw, response=resp, ground_truth=ref, model=judge_model,
    )

    # Visual reasoning (chỉ khi có ảnh)
    visual_score: Optional[float] = None
    if has_image:
        visual_score = await judge_visual_reasoning(
            question=q_raw, response=resp, ground_truth=ref, model=judge_model,
        )

    return {
        "accuracy":     acc,
        "step_clarity": examiner.get("steps"),  # backward-compat alias
        "examiner":     examiner,
        "visual_score": visual_score,
    }


# ── RAGAS (single sample) ──────────────────────────────────────────────────────

def run_ragas_single(sample: dict, judge_model: str) -> dict:
    """
    Chạy RAGAS: context_precision (built-in) + math_context_coverage (custom).
    math_context_coverage thay thế context_recall gốc — phù hợp hơn cho bài Toán.
    """
    try:
        from app.evaluation.ragas_evaluator import RAGASEvaluator
        evaluator = RAGASEvaluator(
            metrics=["context_precision", "math_context_coverage"],
            model=judge_model,
        )
        report = evaluator.run_and_report([sample])
        scores = report.get("summary_scores", {})
        # Đính kèm reason của custom metric để print_question_trace hiển thị
        reasons = report.get("coverage_reasons", [])
        if reasons:
            scores["_math_context_coverage_reason"] = reasons[0]
        return scores
    except Exception as e:
        return {"error": str(e)}


# ── Print trace ────────────────────────────────────────────────────────────────

def print_question_trace(
    q: dict,
    idx: int,
    total: int,
    chunks: list[dict],
    expansion,           # QueryExpansion object
    system_prompt: str,
    response: str,
    model_used: str,
    using_vision: bool,
    judge: dict,
    ragas_scores: Optional[dict],
    verbose_prompt: bool,
):
    qnum   = q.get("question_number", "?")
    qtype  = q.get("type", "")
    chap   = q.get("chapter", "")
    skill  = q.get("skill_id", "")
    correct= q.get("correct_answer", "")

    type_label = {
        "exam_mcq":          "Trắc nghiệm (MCQ)",
        "exam_true_false":   "Đúng / Sai (T/F)",
        "exam_short_answer": "Trả lời ngắn (SA)",
    }.get(qtype, qtype)

    acc     = judge.get("accuracy", 0)
    clarity = judge.get("step_clarity", 0)
    acc_icon = "✅" if acc >= 0.8 else ("⚠️" if acc >= 0.5 else "❌")

    print(f"\n{SEP2}")
    print(f"  CÂU {qnum}  [{idx}/{total}]  —  {type_label}")
    print(f"  Chapter : {chap}")
    print(f"  Skill   : {skill}")
    print(f"  Model   : {model_used}  {'👁 Vision' if using_vision else '📝 Text-only'}")
    print(SEP2)

    # ── Question ──
    print("\n📌 NỘI DUNG CÂU HỎI:")
    print(SEP)
    print(q["question"][:1200])
    if len(q["question"]) > 1200:
        print("  ... [truncated]")

    # ── Query Expansion ──
    print("\n🧠 QUERY EXPANSION (LLM suy luận kiến thức cần thiết):")
    print(SEP)
    if expansion and not expansion.error:
        ch_str  = ", ".join(expansion.chapters)  or "(none)"
        sk_str  = ", ".join(expansion.skill_ids) or "(none)"
        fm_str  = ", ".join(expansion.formula_ids) or "(none)"
        print(f"  📚 Chapters : {ch_str}")
        print(f"  🎯 Skills   : {sk_str}")
        print(f"  📐 Formulas : {fm_str}")
        if expansion.reasoning:
            print(f"  💬 Reasoning: {expansion.reasoning}")
    elif expansion and expansion.error:
        print(f"  ⚠️  Expansion error: {expansion.error} — dùng raw query để RAG")
    else:
        print("  (không có expansion)")

    # ── RAG chunks ──
    print(f"\n🔍 RAG RETRIEVED ({len(chunks)} chunks):")
    print(SEP)
    for i, chunk in enumerate(chunks):
        vs = chunk.get("vector_score", 0)
        bs = chunk.get("bm25_score", 0)
        hs = chunk.get("hybrid_score", 0)
        rs = chunk.get("rerank_score")
        boosted = " 🚀boost" if chunk.get("_boosted") else ""
        meta = chunk.get("metadata", {})
        
        rs_str = f"  rerank={rs:.3f}" if rs is not None else ""
        print(f"  [{i+1}] id={chunk.get('id','?')}  hybrid={hs:.3f}{rs_str}  vec={vs:.3f}  bm25={bs:.3f}{boosted}")
        print(f"       chapter={meta.get('chapter','?')}  skill={meta.get('skill_id','?')}")
        content_preview = chunk.get("content", "")[:300].replace("\n", " ")
        print(f"       > {content_preview}...")
        print()

    # ── System prompt ──
    if verbose_prompt:
        print("\n📋 SYSTEM PROMPT (gửi đến LLM):")
        print(SEP)
        # Show truncated — without the full context (already shown above)
        prompt_lines = system_prompt.split("\n")
        # Print up to line that says TAI LIEU THAM KHAO
        for line in prompt_lines:
            if "TAI LIEU THAM KHAO" in line:
                print(line)
                print("  [... context từ RAG chunks bên trên ...]")
                break
            print(line)
    else:
        print(f"\n📋 SYSTEM PROMPT: [dùng --verbose-prompt để xem đầy đủ]")
        print(f"   Format: {qtype} | CoT: ✅ | RAG framing: ✅")

    # ── Response ──
    print(f"\n🤖 MODEL RESPONSE:")
    print(SEP)
    print(response[:2000])
    if len(response) > 2000:
        print("  ... [truncated — xem file trace để đọc đủ]")

    # ── Scores ──
    print(f"\n📊 SCORES:")
    print(SEP)
    print(f"  Đáp án đúng  : {correct}")
    print(f"  {acc_icon} MathJudge Accuracy        : {acc:.2f}")

    # LLMExaminer rubric
    examiner = judge.get("examiner", {})
    if examiner and not examiner.get("error"):
        ws = examiner.get("weighted_score", 0)
        ws_icon = "✅" if ws >= 0.7 else ("⚠️" if ws >= 0.5 else "❌")
        print(f"\n  🎓 LLM EXAMINER (rubric 4 chiều):")
        print(f"     {ws_icon} Weighted Score       : {ws:.2f}")
        print(f"     {'✅' if examiner.get('accuracy',0)>=0.8 else '❌'} Accuracy (kết quả)  : {examiner.get('accuracy',0):.2f}")
        print(f"     {'✅' if examiner.get('method',0)>=0.7 else '❌'}  Method (phương pháp): {examiner.get('method',0):.2f}")
        print(f"     {'✅' if examiner.get('steps',0)>=0.7 else '❌'}  Steps (trình bày)   : {examiner.get('steps',0):.2f}")
        print(f"     {'✅' if examiner.get('knowledge_use',0)>=0.5 else '❌'} Knowledge use       : {examiner.get('knowledge_use',0):.2f}")
        if examiner.get("comment"):
            print(f"     💬 {examiner['comment']}")
    elif examiner.get("error"):
        print(f"  ⚠️  LLMExaminer error: {examiner['error'][:80]}")

    # Visual reasoning (chỉ hiện khi có ảnh)
    visual = judge.get("visual_score")
    if visual is not None:
        v_icon = "✅" if visual >= 0.7 else ("⚠️" if visual >= 0.5 else "❌")
        print(f"\n  👁  Visual Reasoning Score  : {v_icon} {visual:.2f}")

    # RAGAS retrieval only
    if ragas_scores:
        print(f"\n  📡 RAG RETRIEVAL (RAGAS):")
        if "error" in ragas_scores:
            print(f"     ⚠️  RAGAS error: {ragas_scores['error'][:80]}")
        else:
            for k, v in ragas_scores.items():
                if isinstance(v, float):
                    icon = "✅" if v >= 0.7 else "❌"
                    print(f"     {icon} {k:<28}: {v:.3f}")
            reason = ragas_scores.get("_math_context_coverage_reason")
            if reason:
                print(f"     💬 Reason: {reason}")

    print()



# ── Save trace ─────────────────────────────────────────────────────────────────

def save_trace(trace_records: list[dict], exam: str) -> str:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = EVAL_DIR / f"trace_{exam}_{ts}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for rec in trace_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return str(path)


# ── Main ───────────────────────────────────────────────────────────────────────

async def main(args):
    print(f"""
{SEP2}
  AITutor — Custom Evaluation Monitor
  Exam    : {args.exam.upper()}
  Questions: {args.questions or 'ALL'}
  Types   : {args.types or 'all'}
  RAGAS   : {'OFF' if args.no_ragas else 'ON (per-question)'}
  Judge   : {args.judge_model}
{SEP2}
""")

    # Parse question selection
    question_nums: Optional[set[int]] = None
    if args.questions:
        question_nums = parse_question_selection(args.questions)

    # Parse types
    types_filter: Optional[list[str]] = None
    if args.types:
        types_filter = [t.strip() for t in args.types.split(",")]

    # Load questions
    questions = load_questions(args.exam, question_nums, types_filter)
    if not questions:
        print("❌ Không tìm thấy câu nào phù hợp.")
        sys.exit(1)

    print(f"  ✅ Loaded {len(questions)} câu\n")

    trace_records = []
    all_judges: list[dict] = []

    for idx, q in enumerate(questions, 1):
        # ── Retrieve ──
        chunks, expansion = retrieve_with_trace(
            q["question"], q["skill_id"], q["chapter"], k=args.k
        )
        context_text = "\n\n---\n\n".join(c["content"] for c in chunks) if chunks else "Khong tim thay tai lieu."

        # ── Build prompt ──
        system_prompt = build_system_prompt(context_text, q["type"])

        # ── Generate ──
        img_path = q["image_path"] if q["has_image"] else ""
        response, model_used, using_vision = await generate_with_trace(
            q["question"], system_prompt, q["type"], img_path
        )

        # ── Build sample for judges ──
        sample = {
            "user_input":         q["question"],
            "retrieved_contexts": [c["content"] for c in chunks],
            "response":           response,
            "reference":          q["ground_truth"],
            "_meta": {
                "id":             q["id"],
                "skill_id":       q["skill_id"],
                "chapter":        q["chapter"],
                "type":           q["type"],
                "correct_answer": q["correct_answer"],
                "has_image":      q["has_image"],
                "image_path":     img_path,
            },
        }

        # ── MathJudge ──
        judge_result = await judge_sample_async(sample, args.judge_model)
        all_judges.append(judge_result)

        # ── RAGAS (optional) ──
        ragas_scores: Optional[dict] = None
        if not args.no_ragas:
            ragas_scores = run_ragas_single(sample, args.judge_model)

        # ── Print trace ──
        print_question_trace(
            q=q,
            idx=idx,
            total=len(questions),
            chunks=chunks,
            expansion=expansion,
            system_prompt=system_prompt,
            response=response,
            model_used=model_used,
            using_vision=using_vision,
            judge=judge_result,
            ragas_scores=ragas_scores,
            verbose_prompt=args.verbose_prompt,
        )

        # ── Collect trace record ──
        if args.save_trace:
            trace_records.append({
                "question_number": q.get("question_number"),
                "id":              q["id"],
                "type":            q["type"],
                "skill_id":        q["skill_id"],
                "chapter":         q["chapter"],
                "correct_answer":  q["correct_answer"],
                "model_used":      model_used,
                "using_vision":    using_vision,
                "system_prompt":   system_prompt,
                "question":        q["question"],
                "retrieved_chunks": [
                    {
                        "id":            c.get("id"),
                        "hybrid_score":  c.get("hybrid_score"),
                        "vector_score":  c.get("vector_score"),
                        "bm25_score":    c.get("bm25_score"),
                        "content":       c.get("content", "")[:500],
                        "metadata":      c.get("metadata", {}),
                    }
                    for c in chunks
                ],
                "response":        response,
                "math_accuracy":   judge_result.get("accuracy"),
                "step_clarity":    judge_result.get("step_clarity"),
                "examiner":        judge_result.get("examiner"),
                "visual_score":    judge_result.get("visual_score"),
                "ragas_scores":    ragas_scores,
                "query_expansion": {
                    "chapters":    expansion.chapters,
                    "skill_ids":   expansion.skill_ids,
                    "formula_ids": expansion.formula_ids,
                    "reasoning":   expansion.reasoning,
                    "error":       expansion.error,
                },
            })

    # ── Summary (luôn in, dùng all_judges thay vì trace_records) ──
    print(f"\n{SEP2}")
    print(f"  SUMMARY — {len(questions)} câu")
    print(SEP2)
    if all_judges:
        accs = [j.get("accuracy", 0) for j in all_judges]
        print(f"  Overall Accuracy         : {sum(accs)/len(accs):.3f}  ({sum(1 for a in accs if a >= 0.8)}/{len(accs)} đúng)")
        # LLMExaminer weighted score
        ws_list = [
            j.get("examiner", {}).get("weighted_score")
            for j in all_judges
            if j.get("examiner") and j["examiner"].get("weighted_score") is not None
        ]
        if ws_list:
            print(f"  LLMExaminer Weighted Avg : {sum(ws_list)/len(ws_list):.3f}")
        # Step clarity (from examiner.steps)
        clars = [j.get("step_clarity") for j in all_judges if j.get("step_clarity") is not None]
        if clars:
            print(f"  Step Clarity (steps)     : {sum(clars)/len(clars):.3f}")
        # Visual scores (chỉ Vision questions)
        visuals = [j.get("visual_score") for j in all_judges if j.get("visual_score") is not None]
        if visuals:
            print(f"  Visual Reasoning Avg     : {sum(visuals)/len(visuals):.3f}  ({len(visuals)} Vision questions)")

    # ── Save trace ──
    if args.save_trace and trace_records:
        path = save_trace(trace_records, args.exam)
        print(f"\n  💾 Trace saved → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AITutor Custom Evaluation Monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--exam", default="de01",
        choices=list(EXAM_FILES.keys()),
        help="Đề muốn evaluate (de01/de03/de04/de05)",
    )
    parser.add_argument(
        "--questions", default=None,
        help="Câu cụ thể: '1,3,5-8'. Mặc định: tất cả câu trong đề",
    )
    parser.add_argument(
        "--types", default=None,
        help="Lọc loại: exam_mcq,exam_true_false,exam_short_answer",
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="Số RAG chunks retrieve (mặc định: 5)",
    )
    parser.add_argument(
        "--judge-model", default="gpt-4o-mini",
        help="Model cho MathJudge + RAGAS judge",
    )
    parser.add_argument(
        "--no-ragas", action="store_true",
        help="Bỏ qua RAGAS (chỉ chạy MathJudge — nhanh hơn)",
    )
    parser.add_argument(
        "--verbose-prompt", action="store_true",
        help="In đầy đủ system prompt gửi đến LLM",
    )
    parser.add_argument(
        "--save-trace", action="store_true",
        help="Lưu trace ra file JSONL trong data/evaluation/",
    )

    args = parser.parse_args()
    asyncio.run(main(args))
