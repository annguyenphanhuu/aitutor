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
from typing import Optional, TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils.llm import compatible_temperature

if TYPE_CHECKING:
    from app.rag.query_expander import QueryExpansion

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
    "de08": EXAMS_DIR / "de7plus_de08.json",
    "de09": EXAMS_DIR / "de7plus_de09.json",
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
    from app.rag.query_expander import expand_question
    from app.rag.graph_rag import build_search_queries, merge_search_results

    kb = get_knowledge_base()
    reranker = get_reranker()

    from app.config import get_settings
    settings = get_settings()
    # Tăng pool_k: reranker chỉ có thể giữ lại docs nằm trong pool.
    # Trước đây pool nhỏ có thể làm mất theory_18/19 sau khi soft boost chưa đủ mạnh.
    raw_pool_k = settings.RERANKER_CANDIDATE_K if settings.RERANKER_ENABLED else k * 2
    pool_k = max(raw_pool_k, k * 4)  # ít nhất 20 candidates khi k=5

    # ── Step 1: Query Expansion — LLM suy luận chapter/skill/formula ──────────
    expansion: QueryExpansion = expand_question(question)
    if expansion.error:
        logger.warning("QueryExpander failed: %s — falling back to raw query", expansion.error)

    # ── Step 2: Multi-query retrieval ──────────────────────────────────────────
    # Search cả rewrite học thuật lẫn câu gốc để rewrite giúp bài toán thực tế
    # nhưng không trở thành một điểm lỗi duy nhất của pipeline.
    rewritten_query = expansion.rag_query if not expansion.error else ""
    search_queries = build_search_queries(question, rewritten_query)
    candidates = merge_search_results([
        kb.search_theory(search_query, k=pool_k, alpha=0.55)
        for search_query in search_queries
    ])

    # ── Step 2.5: Inject explicit formulas từ QueryExpansion ───────────────────
    if not expansion.error and expansion.formula_ids:
        explicit_formulas = kb.get_formulas_by_ids(expansion.formula_ids)
        for f in explicit_formulas:
            if not any(c.get("id") == f["id"] for c in candidates):
                candidates.append({
                    "id": f["id"],
                    "content": f["content"],
                    "metadata": f.get("metadata", {}),
                    "hybrid_score": 1.0,  # Điểm tuyệt đối để đảm bảo được rerank
                    "vector_score": 1.0,
                    "bm25_score": 1.0,
                    "_boosted": True
                })

    # ── Step 3: Soft boost (nhẹ hơn) dựa trên QueryExpansion, không dùng metadata đề thi ──
    # Soft boost mạnh hơn để đảm bảo docs đúng skill/chapter luôn vào top pool.
    # Trước: 0.06/0.04 — quá nhẹ, CrossEncoder reranker có thể đảo ngược hoàn toàn.
    # Sau: 0.12/0.07 — có tác động thực sự nhưng vẫn không overwrite semantic score.
    SKILL_BOOST   = 0.12
    CHAPTER_BOOST = 0.07

    expanded_skills   = {s.lower() for s in expansion.skill_ids}
    # Normalize unicode dash variants (en-dash –, em-dash —) → hyphen
    # ChromaDB lưu 'Tổ hợp – Xác suất' (U+2013) nhưng LLM trả về 'Tổ hợp - Xác suất' (U+002D)
    def _norm_ch(s: str) -> str:
        return s.replace("–", "-").replace("—", "-").lower().strip()
    expanded_chapters = {_norm_ch(c) for c in expansion.chapters}

    for r in candidates:
        bonus = 0.0
        meta  = r.get("metadata", {})
        chunk_skill   = str(meta.get("skill_id", "")).lower().strip()
        chunk_chapter = _norm_ch(str(meta.get("chapter",  "")))

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

    ranked = reranker.rerank(rewritten_query or question, candidates, k=k)
    return ranked, expansion


# ── heavy_math classifier ──────────────────────────────────────────────────────

def is_heavy_math(question: str, skill_id: str) -> bool:
    """Return True nếu câu hỏi cần Agentic Tool-Calling (phép tính nặng).

    Tiêu chí:
    - skill_id nằm trong _HEAVY_MATH_SKILLS, HOẶC
    - Câu hỏi chứa từ khóa liên quan đến tích phân/lãi kép/tăng trưởng...
    """
    from app.agents.teacher_agent import requires_math_tool
    return requires_math_tool(question, skill_id)


# ── Build prompt (mirrors dataset_builder.py) ──────────────────────────────────

def build_system_prompt(context: str, question_type: str, use_agentic: bool = False) -> str:
    if question_type == "exam_true_false":
        fmt = (
            "\n\nQUAN TRONG: Cau hoi nay la dang Dung/Sai nhieu menh de. "
            "Bat buoc dat dap an cuoi cung vao the <answer>a-T,b-F,c-T,d-F</answer> voi T la Dung, F la Sai. "
            "Tuyet doi khong dung chu 'Dung/Sai', khong boc trong \\text{} hay bat ky format nao khac."
        )
        if True:  # Always add visual extraction guide for True/False (may have images)
            fmt += (
                "\n\nDAC BIET KHI CO HINH VE (Visual Feature Extraction — BAT BUOC):"
                "\nNeu menh de yeu cau danh gia mot do thi / bang bien thien / hinh ve:"
                "\n  [TRICH XUAT THI GIAC] Truoc khi ket luan Dung/Sai, ban PHAI liet ke CU THE:"
                "\n  1. Phuong trinh duong tiem can (ngang va dung) doc tu hinh."
                "\n  2. Toa do cac diem dac trung: cuc dai, cuc tieu, giao diem truc hoanh, y-intercept."
                "\n  3. Chieu bien thien (ham so tang/giam tren tung khoang)."
                "\n  4. Dang bieu do (hyperbol? parabol? ham bac ba?) va tinh doi xung."
                "\n  KHONG DUOC dung phat bieu 'Neu hinh khong the hien dung thi sai' — DO LA"
                "\n  TRANH TRANH LUAN. Ban PHAI doc hinh va ket luan CHINH XAC."
            )
    elif question_type == "exam_mcq":
        fmt = (
            "\n\nQUAN TRONG: Ban PHAI dat dap an dung vao the <answer>X</answer> voi X la A, B, C hoac D. "
            "Vi du: <answer>A</answer>. Sau do xuong dong va giai thich ngan gon."
        )
    elif question_type == "exam_short_answer":
        fmt = (
            "\n\nQUAN TRONG: Ban PHAI dat con so dap an cuoi cung vao the <answer>GIA_TRI</answer>. "
            "GIA_TRI chi duoc la SO THAP PHAN hoac PHAN SO, tuyet doi KHONG kem theo don vi do luong hay text nao khac. "
            "Dac biet, neu de bai co yeu cau LAM TRON (vi du: 'lam tron den hang phan muoi', 'lam tron den hang phan tram'), "
            "ban PHAI tu thuc hien phep lam tron do va CHI ghi gia tri DA LAM TRON vao the <answer>. "
            "Vi du: <answer>0.56</answer> hay <answer>2</answer>."
        )
    else:
        fmt = ""

    if use_agentic:
        cot = (
            "\n\nQUY TRINH SUY LUAN BAT BUOC — Agentic Tool-Calling:"
            "\n⚠️  NGHIEM CAM: TUYET DOI KHONG duoc tu tinh nham bat ky phep toan nao."
            "\nVoi MOI phep tinh, ban BAT BUOC phai goi Tool (Cong cu SymPy) de may tinh xu ly."
            "\nCac truong hop BAT BUOC phai goi Tool:"
            "\n  - Dao ham: goi compute_derivative"
            "\n  - Tich phan: goi compute_integral"
            "\n  - Giai phuong trinh: goi solve_equation"
            "\n  - Giai bat phuong trinh (>, <, >=, <=): goi solve_inequality"
            "\n  - Tinh logarit, mu, can bac n phuc tap: goi solve_equation hoac simplify_expression"
            "\n  - Tinh gia tri bieu thuc: goi simplify_expression hoac evaluate_at_point"
            "\nQuy trinh:"
            "\n  Buoc 1: Xac dinh dang bai va phuong phap giai."
            "\n  Buoc 2: Thiet lap bieu thuc / phuong trinh / bat phuong trinh can tinh — viet ro ra truoc."
            "\n  Buoc 3: GOI TOOL de tinh. Doi ket qua chinh xac tu SymPy roi moi ket luan."
            "\n  Buoc 4: Neu bai co HINH VE — doc CAN THAN toa do tu hinh truoc khi ket luan."
            "\n  Buoc 5: Bai GTLN/GTNN: goi compute_derivative, solve_equation(f'=0), evaluate_at_point."
            "\n  Buoc 6: Bai tang truong/lai kep: thiet lap bat phuong trinh, goi solve_inequality."
            "\n  Buoc 7: QUY UOC LOG — 'log'/'lg' trong de = log co so 10 => goi log(x, 10) trong SymPy."
        )
    else:
        # Reflection mode: LLM tu suy luan, SymPy chi kiem chung SAU (post-hoc)
        cot = (
            "\n\nQUY TRINH SUY LUAN:"
            "\n  Buoc 1: Xac dinh dang bai, phuong phap giai."
            "\n  Buoc 2: Tinh toan tung buoc ro rang, trinh bay day du."
            "\n  Buoc 3: Neu bai co HINH VE — ghi lai cac toa do / dac diem doc tu hinh truoc khi ket luan."
            "\n  Buoc 4: Kiem tra lai dap an bang dieu kien bien hoac thu nguoc."
            "\n  QUY UOC LOG — 'log'/'lg' trong de bai = log co so 10."
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

_TOOL_CALL_BUDGET = 8  # Số vòng gọi tool tối đa để tránh vòng lặp vô tận


async def _invoke_llm_simple(
    question: str,
    system_prompt: str,
    image_path: str,
    model: str,
) -> tuple[str, bool]:
    """Gọi LLM đơn giản (không tool binding). Trả về (response_text, used_vision)."""
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    from app.config import get_settings
    from app.utils.cost_tracker import log_from_response

    settings = get_settings()
    img_data = _resolve_image(image_path) if image_path else None
    using_vision = img_data is not None

    llm = ChatOpenAI(
        model=model,
        api_key=settings.OPENAI_API_KEY,
        temperature=compatible_temperature(model, 0.1),
    )

    if using_vision:
        b64, mt = img_data
        human_content = [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": f"data:{mt};base64,{b64}", "detail": "high"}},
        ]
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_content)]
    else:
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=question)]

    resp = await llm.ainvoke(messages)
    log_from_response(agent="Solver", model=model, response=resp)
    return resp.content, using_vision


async def generate_with_reflection(
    question: str,
    system_prompt: str,
    image_path: str,
    solver_model: Optional[str] = None,
    question_type: str = "",
) -> tuple[str, str, bool, Optional[str]]:
    """Pipeline bản thường: LLM giải → SymPy Reflection kiểm chứng hậu kỳ.

    Chi phí thấp hơn Agentic vì system prompt gọn, không cần multi-turn tool calls.
    Phù hợp với câu hỏi logic/hình học/khảo sát hàm số.

    Returns (response, model_used, used_vision, reflection_log).
    """
    from app.config import get_settings
    from app.agents.reflection import ReflectionEngine
    from app.utils.answer_format import ensure_answer_tag

    settings = get_settings()
    img_data = _resolve_image(image_path) if image_path else None
    model = solver_model or (settings.VISION_LLM_MODEL if img_data else settings.LLM_MODEL)

    # Step 1: Gọi LLM sinh draft
    draft, using_vision = await _invoke_llm_simple(question, system_prompt, image_path, model)

    # Step 2: SymPy Reflection (post-hoc)
    engine = ReflectionEngine()
    result = await engine.reflect(draft, question)

    # Format reflection log tương tự bản thường cũ
    reflection_log = "\n".join(result.thinking_log)

    final_answer = ensure_answer_tag(result.final_answer, question_type)
    return final_answer, model, using_vision, reflection_log


async def generate_with_trace(
    question: str,
    system_prompt: str,
    question_type: str,
    image_path: str,
    solver_model: Optional[str] = None,
) -> tuple[str, str, bool, Optional[str]]:
    """Sinh câu trả lời dùng Agentic Tool-Calling (SymPy as Function Call).

    Chỉ gọi cho câu hỏi heavy_math. Với câu thường, dùng generate_with_reflection.

    Returns (response, model_used, used_vision, tool_call_log).
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
    from app.config import get_settings
    from app.agents.tools import MATH_TOOL_LIST
    import json

    settings = get_settings()

    img_data = _resolve_image(image_path) if image_path else None
    using_vision = img_data is not None

    if solver_model:
        model = solver_model
    else:
        model = settings.VISION_LLM_MODEL if using_vision else settings.LLM_MODEL

    llm_base = ChatOpenAI(
        model=model,
        api_key=settings.OPENAI_API_KEY,
        temperature=compatible_temperature(model, 0.1),
    )

    # Bind SymPy tools for every solver model. Modern reasoning models support
    # function calling; skipping binding here was the main path that still let
    # them calculate mentally while the trace was labelled "Agentic".
    llm_auto = llm_base.bind_tools(MATH_TOOL_LIST)
    llm_required = llm_base.bind_tools(MATH_TOOL_LIST, tool_choice="required")

    # Xây dựng danh sách messages ban đầu
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

    from app.utils.cost_tracker import log_from_response

    # ── Agentic Tool-Calling Loop ─────────────────────────────────────────────
    tool_call_log_lines: list[str] = ["🔧 Bắt đầu giải với Agentic SymPy Tool-Calling..."]
    tool_map = {t.name: t for t in MATH_TOOL_LIST}
    final_response = ""
    executed_tool = False

    for turn in range(_TOOL_CALL_BUDGET + 1):
        llm = llm_required if not executed_tool else llm_auto
        resp = await llm.ainvoke(messages)
        log_from_response(agent="Solver", model=model, response=resp)

        # Kiểm tra LLM có muốn gọi tool không
        tool_calls = getattr(resp, "tool_calls", None)

        if not tool_calls:
            if not executed_tool:
                if turn < _TOOL_CALL_BUDGET:
                    messages.extend([
                        resp,
                        HumanMessage(content="Bạn chưa gọi SymPy. Hãy gọi tool trước khi kết luận."),
                    ])
                    tool_call_log_lines.append("⚠️ Từ chối draft tự tính vì chưa có tool call.")
                    continue
                final_response = "Không thể xác minh phép tính vì mô hình chưa gọi công cụ SymPy."
                tool_call_log_lines.append("❌ Dừng: đã hết lượt nhưng chưa có tool call nào.")
                break
            # Không còn tool call → đây là đáp án cuối cùng
            final_response = resp.content
            tool_call_log_lines.append(f"✅ Hoàn tất sau {turn} lượt gọi tool.")
            break

        # LLM muốn gọi tool → thực thi từng tool
        messages.append(resp)  # Thêm AI message (chứa tool_calls) vào history
        tool_call_log_lines.append(f"\n🔁 Lượt {turn + 1}: LLM yêu cầu {len(tool_calls)} tool call(s):")

        for tc in tool_calls:
            tool_name = tc.get("name", "")
            tool_args = tc.get("args", {})
            tool_call_id = tc.get("id", f"call_{turn}")

            tool_fn = tool_map.get(tool_name)
            if tool_fn is None:
                tool_result_str = f"ERROR: Tool '{tool_name}' không tồn tại."
            else:
                try:
                    tool_result = tool_fn.invoke(tool_args)
                    tool_result_str = (
                        tool_result
                        if isinstance(tool_result, str)
                        else json.dumps(tool_result, ensure_ascii=False)
                    )
                except Exception as e:
                    tool_result_str = f"ERROR khi chạy {tool_name}: {e}"

            tool_call_log_lines.append(
                f"  📐 [{tool_name}] args={json.dumps(tool_args, ensure_ascii=False)}"
                f"\n     → {tool_result_str}"
            )

            # Đưa kết quả tool vào messages để LLM đọc
            messages.append(ToolMessage(
                content=tool_result_str,
                tool_call_id=tool_call_id,
            ))
            executed_tool = True

    else:
        # Đã hết budget mà LLM vẫn gọi tool: không dựng một đáp án chưa được tổng hợp.
        tool_call_log_lines.append(f"⚠️ Đã đạt giới hạn {_TOOL_CALL_BUDGET} lượt gọi tool mà chưa có kết luận.")
        final_response = "Đã xác minh phép tính bằng SymPy nhưng chưa thể tổng hợp đáp án trong giới hạn lượt gọi."

    from app.utils.answer_format import ensure_answer_tag
    final_response = ensure_answer_tag(final_response, question_type)
    tool_call_log = "\n".join(tool_call_log_lines)
    return final_response, model, using_vision, tool_call_log


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

    # LLMExaminer (4-criterion rubric):
    # - accuracy is injected deterministically from MathJudge (never re-derived by LLM)
    # - method/steps/knowledge_use are scored by the LLM
    examiner = await judge_examiner(
        question=q_raw,
        response=resp,
        ground_truth=ref,
        model=judge_model,
        correct_answer=correct,
        question_type=qtype,
        deterministic_accuracy=acc,  # inject MathJudge score — bypasses LLM accuracy
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
    reflection_log: Optional[str],
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
        prompt_lines = system_prompt.split("\n")
        for line in prompt_lines:
            if "TAI LIEU THAM KHAO" in line:
                print(line)
                print("  [... context từ RAG chunks bên trên ...]")
                break
            print(line)
    else:
        print("\n📋 SYSTEM PROMPT: [dùng --verbose-prompt để xem đầy đủ]")
        print(f"   Format: {qtype} | CoT: ✅ | RAG framing: ✅")

    # -- SymPy Log (Agentic Tool Calls or Reflection) --
    if reflection_log:
        print("\n⚙️  SYM-PY REFLECTION:")
        print(SEP)
        safe_log = reflection_log.encode("utf-8", errors="replace").decode("utf-8")
        try:
            print(safe_log)
        except UnicodeEncodeError:
            print(safe_log.encode("ascii", errors="replace").decode("ascii"))

    # ── Response ──
    print("\n🤖 MODEL RESPONSE:")
    print(SEP)
    print(response[:2000])
    if len(response) > 2000:
        print("  ... [truncated — xem file trace để đọc đủ]")

    # ── Scores ──
    print("\n📊 SCORES:")
    print(SEP)
    print(f"  Đáp án đúng  : {correct}")
    print(f"  {acc_icon} MathJudge Accuracy        : {acc:.2f}")

    # LLMExaminer rubric
    examiner = judge.get("examiner", {})
    if examiner and not examiner.get("error"):
        ws = examiner.get("weighted_score", 0)
        ws_icon = "✅" if ws >= 0.7 else ("⚠️" if ws >= 0.5 else "❌")
        print("\n  🎓 LLM EXAMINER (rubric 4 chiều):")
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
        print("\n  📡 RAG RETRIEVAL (RAGAS):")
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

class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        try:
            self.terminal.write(message)
        except UnicodeEncodeError:
            # Windows terminal may not support all Unicode; fall back to ASCII
            self.terminal.write(message.encode("ascii", errors="replace").decode("ascii"))
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

async def main(args):
    if args.log_file:
        # Tự động tạo thư mục chứa log nếu chưa có
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
        sys.stdout = TeeLogger(args.log_file)

    print(f"""
{SEP2}
  AITutor — Custom Evaluation Monitor
  Exam    : {args.exam.upper()}
  Questions: {args.questions or 'ALL'}
  Types   : {args.types or 'all'}
  RAGAS   : {'OFF' if args.no_ragas else 'ON (per-question)'}
  Judge   : {args.judge_model}
  Log File: {args.log_file or 'None'}
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
        clean_q = clean_question_for_eval(q["question"])

        # ── Retrieve ──
        chunks, expansion = retrieve_with_trace(
            clean_q, q["skill_id"], q["chapter"], k=args.k
        )
        context_text = "\n\n---\n\n".join(c["content"] for c in chunks) if chunks else "Khong tim thay tai lieu."

        # ── Classify: heavy_math → Agentic, else → Reflection ──
        use_agentic = is_heavy_math(clean_q, q["skill_id"])
        # ── Build prompt ──
        system_prompt = build_system_prompt(context_text, q["type"], use_agentic=use_agentic)

        # ── Generate ──
        img_path = q["image_path"] if q["has_image"] else ""
        if use_agentic:
            response, model_used, using_vision, reflection_log = await generate_with_trace(
                q["question"], system_prompt, q["type"], img_path, args.solver_model,
            )
        else:
            response, model_used, using_vision, reflection_log = await generate_with_reflection(
                q["question"],
                system_prompt,
                img_path,
                args.solver_model,
                question_type=q["type"],
            )

        # ── Build sample for judges ──
        sample = {
            "user_input":         clean_q,
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
            reflection_log=reflection_log,
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
                "reflection_log":  reflection_log,
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
            
        from app.utils.cost_tracker import session_summary
        print(f"\n  {session_summary()}")

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
        "--solver-model", default=None,
        help="Model giải toán. Nếu không truyền sẽ lấy từ config.py (LLM_MODEL/VISION_LLM_MODEL)",
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
    parser.add_argument(
        "--use-reflection", action="store_true",
        help="Bật luồng Reflection để dùng SymPy kiểm chứng tính toán (mặc định: tắt)",
    )
    parser.add_argument(
        "--log-file", default=None,
        help="Đường dẫn file txt/log để lưu lại kết quả in ra màn hình (ví dụ: data/evaluation/log_de01.txt)",
    )

    args = parser.parse_args()
    asyncio.run(main(args))
