"""Các node của tutor graph + hàm routing thuần.

``TutorNodes`` giữ các agent hiện có (Teacher/Planner/Visualizer) làm worker —
graph chỉ thay thế control flow if/elif của Orchestrator cũ. Test patch các
class tại ``app.graph.nodes.*`` trước khi khởi tạo (giống pattern
test_orchestrator.py patch ``app.agents.orchestrator.*``).

Runtime handles đi qua ``config["configurable"]``:
  db (AsyncSession), user_id (int), session_id (Optional[int]),
  trace_id (Optional[str]), streaming (bool).

LƯU Ý: các node PHẢI chạy tuần tự (builder.py nối edge thẳng hàng) —
AsyncSession của SQLAlchemy không dùng đồng thời được.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.types import StreamWriter
from openai import AsyncOpenAI

from app.agents.assessor_agent import AssessorAgent
from app.agents.context_utils import format_formulas, format_skill_names, normalize_skill_ids
from app.agents.contracts import (
    GradingExtraction,
    IntentClassification,
    PedagogyAssessment,
)
from app.agents.grading.extractor import GradingExtractor
from app.agents.grading.policy import decide, reconcile_assessment
from app.agents.grading.verifier import verify
from app.agents.planner_agent import PlannerAgent
from app.agents.teacher_agent import TeacherAgent
from app.agents.visualizer_agent import VisualizerAgent
from app.config import get_settings
from app.db.session_state import get_session_state, upsert_session_state
from app.graph import formatters
from app.graph.state import SessionSnapshot, TutorState
from app.knowledge_tracing.bkt import BKTModel
from app.knowledge_tracing.service import get_all_masteries, get_mastery_profile, identify_gaps, update_mastery
from app.knowledge_tracing.skill_graph import SKILLS
from app.utils.answer_format import strip_answer_tag_for_chat
from app.utils.cost_tracker import log_from_response
from app.utils.llm import compatible_temperature

logger = logging.getLogger(__name__)
settings = get_settings()

# Intent không stream token: node tính blocking, finalize emit 1 token frame + done giàu metadata
NON_STREAM_INTENTS = frozenset(
    {"off_topic", "plan", "quiz", "review", "diagnostic", "visualize", "assess",
     "greeting", "motivation"}
)

CLASSIFIER_SYSTEM_PROMPT = """Phân loại ý định của học sinh thành một trong các loại sau:
- "explain": Hỏi bài, cần giải thích lý thuyết hoặc hướng dẫn giải từng bước
- "answer": Chỉ muốn biết đáp án / kết quả của bài toán (ví dụ: "đáp án là gì?", "kết quả bằng bao nhiêu?", "cho tôi xem đáp án")
- "assess": Gửi câu trả lời CỦA CHÍNH MÌNH để kiểm tra đúng/sai
- "plan": Hỏi về kế hoạch học tập, nên ôn gì
- "quiz": Muốn làm bài kiểm tra, luyện tập, ra đề (ví dụ: "cho em làm quiz", "ra đề thi đạo hàm", "cho bài tập luyện tập")
- "review": Muốn ôn bài cũ, ôn tập lại (ví dụ: "ôn tập", "nhắc lại kiến thức cũ", "có gì cần ôn không")
- "diagnostic": Muốn kiểm tra đầu vào, đánh giá năng lực (ví dụ: "test đầu vào", "đánh giá năng lực", "kiểm tra trình độ")
- "visualize": Yêu cầu CHÍNH là VẼ/hiển thị đồ thị hoặc hình minh họa (ví dụ: "vẽ đồ thị y = x²", "minh họa hình học cho em xem").
  LƯU Ý: nếu học sinh HỎI về tính chất hoặc đọc thông tin từ đồ thị (đồng biến/nghịch biến,
  cực trị, giao điểm, tiệm cận...) — kể cả khi tin nhắn có chữ "đồ thị" — thì đó là "explain", KHÔNG phải "visualize".
- "greeting": Chào hỏi, giới thiệu bản thân, làm quen, hoặc nhờ gia sư giúp học Toán một cách chung chung
  chưa có bài toán cụ thể (ví dụ: "chào thầy", "em tên Minh, em mất gốc toán, thầy giúp em với")
- "motivation": Tâm sự chán nản, lo lắng, áp lực về việc học Toán hoặc thi cử, cần được động viên
  (ví dụ: "em sợ trượt tốt nghiệp", "em học dốt toán quá, nản lắm rồi")
- "off_topic": Câu hỏi/yêu cầu KHÔNG liên quan đến Toán học lẫn việc học (ví dụ: tư vấn tình cảm, hỏi thời tiết,
  chuyện phiếm, nhờ dạy môn khác, yêu cầu làm việc khác ngoài Toán). Chào hỏi → "greeting"; tâm sự về việc học → "motivation".

QUY TẮC QUAN TRỌNG về TIN NHẮN TIẾP NỐI (đọc kỹ LỊCH SỬ HỘI THOẠI ở trên):
Tin nhắn ngắn tách khỏi ngữ cảnh thường vô nghĩa. Nếu lịch sử cho thấy hai thầy trò
ĐANG DỞ một bài Toán, thì các tin nhắn sau đây thuộc về bài đó — KHÔNG phải chào hỏi,
KHÔNG phải tâm sự, KHÔNG phải lạc đề:
- Học sinh báo bí ở bước đang làm ("em không biết làm ạ", "em chịu rồi", "vẫn không hiểu",
  "bó tay thầy ơi") → intent="explain" (thầy sẽ gỡ tiếp bước đang vướng, KHÔNG phải an ủi suông).
- Học sinh phản biện / không đồng ý với thầy ("thầy sai rồi", "em không đồng ý",
  "sách em ghi khác mà", "em vẫn nghĩ là ...") → intent="explain" (đây là tranh luận Toán học
  cần được giải thích tới nơi, TUYỆT ĐỐI KHÔNG phải "off_topic").
- Học sinh trả lời câu hỏi gợi mở của thầy bằng một kết quả ngắn ("bằng 2 ạ", "là 3x^2",
  "m = 2") → intent="assess", is_answer_submission=true.
- Học sinh xin quay lại bài cũ ("quay lại bài lúc nãy", "làm tiếp thế nào ạ") → intent="explain".
Chỉ dùng "greeting"/"motivation"/"off_topic" khi tin nhắn THỰC SỰ rời khỏi bài Toán đang làm
(chào hỏi lúc mở đầu, tâm sự về chuyện học nói chung, hỏi chuyện ngoài Toán).

QUY TẮC QUAN TRỌNG về is_answer_submission:
- is_answer_submission = true CHỈ KHI tin nhắn chứa lời giải hoặc đáp án do CHÍNH học sinh làm ra
  (thường ở ngôi thứ nhất: "em tính được...", "em giải ra...", "đáp án của em là...").
- Học sinh THUẬT LẠI lời thầy/cô/bạn/sách rồi hỏi đúng sai KHÔNG phải nộp bài
  → intent="explain", is_answer_submission=false.
- Tin nhắn chỉ nhắc lại đề bài hoặc hỏi cách làm → is_answer_submission=false.

Ví dụ phân loại:
- "Thầy em bảo rằng ∫x²dx = x³/2 + C. Thầy em đúng chứ?" → intent="explain", is_answer_submission=false
  (xác minh mệnh đề của người khác, không phải bài làm của học sinh)
- "Bạn em nói đạo hàm của sin(x) là -cos(x), đúng không?" → intent="explain", is_answer_submission=false
- "Em giải ra ∫x²dx = x³/3 + C, đúng chưa ạ?" → intent="assess", is_answer_submission=true
- "Đáp án của em là x = 2 và x = -2" → intent="assess", is_answer_submission=true
- "Vẽ đồ thị hàm số y = x³ - 3x + 2 giúp em" → intent="visualize"
- "Nhìn đồ thị đó thì hàm số đồng biến trên khoảng nào ạ?" → intent="explain"
  (hỏi về tính chất đọc từ đồ thị, không phải yêu cầu vẽ)
- "Chào thầy ạ!" → intent="greeting"
- "Em bị mất gốc toán, thầy giúp em được không?" → intent="greeting"
- "Em nản quá, chắc trượt tốt nghiệp mất thầy ơi" → intent="motivation"
- "Thầy ơi dạy em tiếng Anh với" → intent="off_topic"
- (đang dở bài tích phân) "Em không biết làm ạ" → intent="explain"
  (báo bí giữa bài, KHÔNG phải greeting)
- (thầy vừa nói đạo hàm sin(2x) là 2cos(2x)) "Thầy sai rồi, em không đồng ý" → intent="explain"
  (phản biện Toán học, KHÔNG phải off_topic)
- (thầy vừa hỏi "nghiệm của 2x+4=0 là bao nhiêu?") "Bằng -2 ạ"
  → intent="assess", is_answer_submission=true

Đồng thời xác định:
- skill_ids / skill_id: các kỹ năng Toán 12 liên quan
- formula_ids: công thức liên quan
- contains_problem: true nếu tin nhắn chứa một đề bài toán cụ thể
- confidence: độ tin cậy phân loại (0.0-1.0)

Danh sách skill_id:
{skills_list}

Danh sách formula_id:
{formulas_list}
"""


def route_intent(state: TutorState) -> str:
    """Cạnh điều kiện sau hydrate — pure function, unit-test được."""
    intent = state.get("intent", "explain")
    classification = state.get("classification")
    if intent == "assess" and classification and classification.is_answer_submission:
        return "grade_extract"
    # Học sinh bí / phản biện giữa bài mà bị gán nhãn rời bài → kéo về teach,
    # nơi có ràng buộc Socratic và chỉ thị worked micro-step (F-01).
    if should_stay_on_problem(state):
        logger.info("Guard: intent=%s giữa bài đang dở → teach", intent)
        return "teach"
    if intent in ("greeting", "motivation"):
        return "social"
    if intent in ("off_topic", "plan", "quiz", "review", "diagnostic", "visualize"):
        return intent
    return "teach"


def route_gradable(state: TutorState) -> str:
    """Cạnh điều kiện sau grade_extract: abstain → teach (giải thích thay vì chấm)."""
    extraction = state.get("extraction")
    return "grade_verify" if extraction and extraction.gradable else "teach"


def is_assessment_output(content: str) -> bool:
    """Nhận diện score card cũ do engine sinh ra (port từ Orchestrator)."""
    head = content.lstrip()[:40]
    return head.startswith(("✅", "❌")) and "Điểm:" in head


def find_original_question(chat_history: list[dict], fallback: str) -> str:
    """Tìm đề bài gốc trong history, bỏ qua các score card cũ (port từ Orchestrator)."""
    for h in reversed(chat_history or []):
        if h.get("role") != "assistant":
            continue
        content = h.get("content") or ""
        if is_assessment_output(content):
            continue
        return content
    return fallback


# ── F-01: phát hiện học sinh bí lặp lại cùng một bước ────────────────────
# Chính sách "bí lần 2 → giải chi tiết bước đó" trước đây CHỈ dựa vào LLM đọc
# lịch sử. Trong luồng stream, reasoning_effort=low khiến LLM hay bỏ qua luật
# này → học sinh mắc kẹt. Ta đếm số lượt bí liên tiếp một cách xác định rồi
# ép một "worked micro-step" khi chạm ngưỡng.
_STUCK_PATTERNS = (
    "không biết", "ko biết", "khong biet", "k biết", "kg biết",
    "chưa biết", "chưa hiểu", "chua hieu", "không hiểu", "khong hieu",
    "vẫn bí", "van bi", "vẫn không", "van khong", "vẫn chưa",
    "bí quá", "bí rồi", "chịu", "bó tay", "không làm được",
    "không nghĩ ra", "chưa nghĩ ra", "nghĩ không ra", "không ra",
    "giúp em với", "chỉ em với", "làm sao", "sao làm",
)

# Ngưỡng bắt buộc render worked micro-step (bí lần thứ 2 ở cùng bước)
STUCK_THRESHOLD = 2


def _is_stuck_message(text: str) -> bool:
    """True nếu tin nhắn thể hiện học sinh đang bí / không biết làm tiếp."""
    if not text:
        return False
    low = text.lower()
    return any(p in low for p in _STUCK_PATTERNS)


def count_consecutive_stuck(chat_history: list[dict], current_message: str) -> int:
    """Đếm số lượt HỌC SINH bí liên tiếp tính đến tin nhắn hiện tại.

    Đi ngược lịch sử, bỏ qua lượt assistant; dừng khi gặp một lượt học sinh
    KHÔNG phải là câu 'bí'. Pure function — unit-test được.
    """
    if not _is_stuck_message(current_message):
        return 0
    count = 1
    for h in reversed(chat_history or []):
        role = h.get("role")
        if role == "assistant":
            continue
        if role == "user":
            if _is_stuck_message(h.get("content") or ""):
                count += 1
            else:
                break
    return count


STUCK_DIRECTIVE = (
    "⚠️ CHỈ THỊ SƯ PHẠM (ưu tiên cao nhất, GHI ĐÈ luật 'chỉ gợi ý'): "
    "Học sinh đã nói không biết / vẫn bí {n} lần LIÊN TIẾP ở cùng một bước. "
    "NGỪNG chỉ đưa gợi ý mơ hồ. Bây giờ BẮT BUỘC giải chi tiết TRỌN VẸN đúng "
    "MỘT bước đang vướng: thay số THẬT của đề vào, viết ra kết quả CỤ THỂ của "
    "riêng bước đó (tuyệt đối KHÔNG để ô '?', KHÔNG để trống, KHÔNG nói chung "
    "chung), và giải thích ngắn gọn vì sao làm vậy. Sau đó hỏi MỘT câu gợi mở "
    "cho bước KẾ TIẾP mà em chưa làm. Vẫn KHÔNG tiết lộ đáp án cuối của cả bài."
)


def build_stuck_directive(stuck_count: int) -> Optional[str]:
    """Chỉ thị runtime ép worked micro-step khi học sinh bí >= ngưỡng.

    Trả về None nếu chưa tới ngưỡng (không thêm gì vào prompt).
    """
    if stuck_count < STUCK_THRESHOLD:
        return None
    return STUCK_DIRECTIVE.format(n=stuck_count)


# ── Ngữ cảnh cho classifier ──────────────────────────────────────────────
# Classifier trước đây chỉ thấy tin nhắn hiện tại nên đoán bừa các tin nhắn
# tiếp nối ("em không biết làm", "thầy sai rồi", "bằng 2 ạ") với confidence
# 0.93-0.98 → route sang social/off_topic, rơi khỏi luồng dạy.
CLASSIFIER_HISTORY_TURNS = 4
_CLASSIFIER_HISTORY_CHARS = 400

_THINKING_RE = re.compile(r"<thinking>.*?</thinking>", flags=re.DOTALL)


def _history_messages_for_classifier(chat_history: Optional[list[dict]]) -> list:
    """Vài lượt gần nhất, dạng message LangChain, cho classifier đọc ngữ cảnh.

    Bỏ khối <thinking> của assistant (log kiểm chứng nội bộ, chỉ gây nhiễu) và
    cắt ngắn từng lượt — classifier chỉ cần biết hai thầy trò đang làm bài gì.
    """
    messages = []
    for turn in (chat_history or [])[-CLASSIFIER_HISTORY_TURNS:]:
        content = _THINKING_RE.sub("", turn.get("content") or "").strip()
        if not content:
            continue
        content = content[:_CLASSIFIER_HISTORY_CHARS]
        if turn.get("role") == "user":
            messages.append(HumanMessage(content=content))
        elif turn.get("role") == "assistant":
            messages.append(AIMessage(content=content))
    return messages


# ── Giữ học sinh ở lại bài đang dở ───────────────────────────────────────
_CHALLENGE_PATTERNS = (
    "thầy sai", "thay sai", "sai rồi", "sai roi", "không đồng ý", "khong dong y",
    "em không nghĩ vậy", "em nghĩ khác", "em vẫn nghĩ", "van nghi",
    "sách em ghi", "sach em ghi", "sách ghi", "cô em bảo", "thầy em bảo",
    "không đúng", "khong dung", "em phản đối",
)

_STAY_ON_PROBLEM_INTENTS = frozenset({"greeting", "motivation", "off_topic"})


def _is_challenge_message(text: str) -> bool:
    """True nếu học sinh đang phản biện / không đồng ý với thầy."""
    if not text:
        return False
    low = text.lower()
    return any(p in low for p in _CHALLENGE_PATTERNS)


def should_stay_on_problem(state: TutorState) -> bool:
    """True khi intent 'rời bài' nhưng học sinh thực ra đang bí/phản biện giữa bài.

    Guard hẹp có chủ ý: chỉ cứu hai loại tin nhắn tiếp nối đã biết là hay bị
    phân loại nhầm, và chỉ khi session còn một bài đang dở. Câu lạc đề thật
    ("tối nay đá bóng đội nào thắng") vẫn đi đúng vào off_topic. Pure function.
    """
    if state.get("intent") not in _STAY_ON_PROBLEM_INTENTS:
        return False
    snapshot = state.get("session_state")
    if snapshot is None or not snapshot.current_problem:
        return False
    message = state.get("message") or ""
    return _is_stuck_message(message) or _is_challenge_message(message)


# ── F-02: đại số nền không được gắn nhãn kỹ năng giải tích ────────────────
# Skill graph chỉ có 6 nhóm kiến thức Toán 12; không có PT bậc hai / biến đổi
# đại số nền. Classifier hay chọn nhãn gần nhất (vd derivative_basic cho
# "giải x^2-5x+6=0") → sai mastery/BKT/RAG. Guard xác định: nếu tin nhắn là
# đại số nền thuần và KHÔNG có ngữ cảnh giải tích thì gỡ nhãn giải tích.
_CALCULUS_CHAPTERS = frozenset({"Đạo hàm", "Nguyên hàm và Tích phân"})

_CALCULUS_CONTEXT_RE = re.compile(
    r"đạo hàm|nguyên hàm|tích phân|∫|cực trị|cực đại|cực tiểu|đơn điệu|"
    r"đồng biến|nghịch biến|tiếp tuyến|giới hạn|\blim\b|tiệm cận|khảo sát|"
    r"biến thiên|f\s*'|y\s*'",
    flags=re.IGNORECASE,
)

_QUADRATIC_RE = re.compile(r"x\s*(?:\^|\*\*)\s*2|x²|x2\b")

_SOLVE_HINTS = (
    "giải phương trình", "giải pt", "phương trình bậc", "tìm nghiệm",
    "nghiệm của", "tìm x", "giải bất phương trình",
)


def _is_calculus_skill(skill_id: Optional[str]) -> bool:
    info = SKILLS.get(skill_id) if skill_id else None
    return bool(info and info.get("chapter") in _CALCULUS_CHAPTERS)


def is_foundation_algebra(message: str) -> bool:
    """True nếu tin nhắn là bài đại số NỀN (giải PT bậc nhất/bậc hai) và KHÔNG
    kèm ngữ cảnh giải tích. Loại này không thuộc 6 chương Toán 12 → không nên
    map sang kỹ năng đạo hàm/tích phân. Pure function — unit-test được.
    """
    if not message:
        return False
    if _CALCULUS_CONTEXT_RE.search(message):
        return False
    has_eq = "=" in message
    quadratic = bool(_QUADRATIC_RE.search(message))
    solving = any(h in message.lower() for h in _SOLVE_HINTS)
    if quadratic and (has_eq or solving):
        return True
    if solving and has_eq and re.search(r"\bx\b", message, flags=re.IGNORECASE):
        return True
    return False


def _cfg(config: RunnableConfig) -> dict:
    return config.get("configurable", {}) or {}


def _writer_if_streaming(config: RunnableConfig, writer: StreamWriter):
    """Trả writer khi request là SSE stream — LangGraph inject writer qua tham số node."""
    return writer if _cfg(config).get("streaming") else None


class TutorNodes:
    """Container các node — giữ agent singletons, patchable trong test."""

    def __init__(self):
        self.teacher = TeacherAgent()
        self.planner = PlannerAgent()
        self.visualizer = VisualizerAgent()
        self.bkt = BKTModel()
        self.extractor = GradingExtractor()
        self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

        mini_temp = compatible_temperature(settings.LLM_MODEL_MINI, 0.1)
        self.classifier_llm = ChatOpenAI(
            model=settings.LLM_MODEL_MINI,
            api_key=settings.OPENAI_API_KEY,
            temperature=mini_temp,
        ).with_structured_output(
            IntentClassification, method="function_calling", include_raw=True
        )
        self.feedback_llm = ChatOpenAI(
            model=settings.LLM_MODEL_MINI,
            api_key=settings.OPENAI_API_KEY,
            temperature=mini_temp,
        ).with_structured_output(
            PedagogyAssessment, method="function_calling", include_raw=True
        )
        # Node social (greeting/motivation): text tự nhiên, cần ấm áp → temp cao hơn
        self.social_llm = ChatOpenAI(
            model=settings.LLM_MODEL_MINI,
            api_key=settings.OPENAI_API_KEY,
            temperature=compatible_temperature(settings.LLM_MODEL_MINI, 0.7),
        )

    # ── classify ─────────────────────────────────────────────────────────

    async def classify(self, state: TutorState, config: RunnableConfig) -> dict:
        from app.utils.langfuse_client import end_generation, new_generation

        message = state["message"]
        skills_list = "\n".join(
            f"- {k}: {v['name']} ({v['chapter']})" for k, v in SKILLS.items()
        )
        try:
            from app.rag.formula_registry import list_formulas

            formulas_list = "\n".join(
                f"- {formula['id']}: {formula.get('metadata', {}).get('chapter', '')}"
                for formula in list_formulas()
            )
        except Exception:
            formulas_list = ""

        system_prompt = CLASSIFIER_SYSTEM_PROMPT.format(
            skills_list=skills_list, formulas_list=formulas_list
        )

        gen = new_generation(
            name="graph.classify_intent",
            model=settings.LLM_MODEL_MINI,
            input_text=message[:300],
            trace_id=_cfg(config).get("trace_id"),
        )

        classification = IntentClassification()  # fail-safe: explain
        input_tokens = output_tokens = 0
        try:
            output = await self.classifier_llm.ainvoke([
                SystemMessage(content=system_prompt),
                *_history_messages_for_classifier(state.get("chat_history")),
                HumanMessage(content=f"Tin nhắn học sinh: {message}"),
            ])
            raw = output.get("raw")
            if raw is not None:
                log_from_response(
                    agent="Classifier", model=settings.LLM_MODEL_MINI, response=raw
                )
                usage = getattr(raw, "usage_metadata", None) or {}
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
            if output.get("parsed") is not None:
                classification = output["parsed"]
        except Exception as exc:
            logger.warning("Classifier lỗi, fallback intent=explain: %s", exc)

        end_generation(
            gen,
            output=classification.model_dump_json(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        # Lọc skill_id theo registry — LLM đôi khi bịa skill_id không tồn tại
        # (vd "giải phương trình bậc hai"); để lọt sẽ sinh rác trong SkillMastery.
        skill_ids = [s for s in classification.resolved_skill_ids() if s in SKILLS]
        skill_id = classification.skill_id if classification.skill_id in SKILLS else None
        skill_id = skill_id or (skill_ids[0] if skill_ids else None)

        # F-02: đại số nền (PT bậc hai...) KHÔNG được gắn nhãn giải tích.
        # Gỡ nhãn sai → skill_id=None (kiến thức nền chưa có trong graph) để
        # không cập nhật nhầm mastery/BKT cho đạo hàm/tích phân.
        if is_foundation_algebra(message) and _is_calculus_skill(skill_id):
            logger.info(
                "F-02 guard: gỡ nhãn giải tích '%s' cho tin nhắn đại số nền", skill_id
            )
            skill_ids = [s for s in skill_ids if not _is_calculus_skill(s)]
            skill_id = skill_ids[0] if skill_ids else None
        return {
            "classification": classification,
            "intent": classification.intent,
            "skill_id": skill_id,
            "skill_ids": skill_ids,
            "formula_ids": classification.formula_ids,
        }

    # ── hydrate ──────────────────────────────────────────────────────────

    async def hydrate(
        self, state: TutorState, config: RunnableConfig, writer: StreamWriter = None
    ) -> dict:
        cfg = _cfg(config)
        db = cfg["db"]
        user_id = cfg.get("user_id", 1)
        session_id = cfg.get("session_id")

        masteries = await get_all_masteries(db, user_id)

        snapshot = SessionSnapshot()
        if session_id is not None:
            row = await get_session_state(db, session_id)
            if row is not None:
                snapshot = SessionSnapshot(
                    current_problem=row.current_problem,
                    awaiting_answer=bool(row.awaiting_answer),
                    last_skill_ids=row.last_skill_ids or [],
                )

        # F-02/F-03: follow-up không tự nhận diện được skill (classifier trả None
        # cho "vậy bước tiếp theo?", "vẫn không biết"...) → kế thừa skill của bài
        # đang giải trong session thay vì để null (meta hiển thị đúng skill, RAG
        # và mastery bám đúng bài).
        skill_id = state.get("skill_id")
        skill_ids = state.get("skill_ids") or []
        inherited = False
        if skill_id is None and snapshot.last_skill_ids:
            skill_ids = list(snapshot.last_skill_ids)
            skill_id = skill_ids[0]
            inherited = True

        current_mastery = masteries.get(skill_id, 0.1) if skill_id else 0.1

        requested_mode = state.get("requested_mode", "auto")
        mode = (
            ("exam" if current_mastery >= 0.7 else "socratic")
            if requested_mode == "auto"
            else requested_mode
        )

        writer = _writer_if_streaming(config, writer)
        if writer is not None:
            skill_info = SKILLS.get(skill_id, {})
            writer(json.dumps({
                "type": "meta",
                "skill_id": skill_id,
                "skill_ids": skill_ids,
                "formula_ids": state.get("formula_ids") or [],
                "skill_name": skill_info.get("name"),
                "mastery_level": round(current_mastery, 3),
                "mode_used": mode,
                "intent": state.get("intent"),
            }, ensure_ascii=False))

        result = {
            "masteries": masteries,
            "current_mastery": current_mastery,
            "mastery_level": self.bkt.get_mastery_level(current_mastery),
            "mode": mode,
            "session_state": snapshot,
        }
        if inherited:
            result["skill_id"] = skill_id
            result["skill_ids"] = skill_ids
        return result

    # ── các intent tĩnh ──────────────────────────────────────────────────

    async def off_topic(self, state: TutorState, config: RunnableConfig) -> dict:
        return {
            "response_text": formatters.OFF_TOPIC_TEXT,
            "mode_used": "off_topic",
            "result": {
                "response": formatters.OFF_TOPIC_TEXT,
                "skill_id": None,
                "skill_name": None,
                "mastery_level": None,
                "mode_used": "off_topic",
            },
        }

    # ── social (greeting / motivation) ───────────────────────────────────

    SOCIAL_SYSTEM_PROMPT = """Bạn là "thầy" — gia sư Toán 12 thân thiện, tận tâm của học sinh.
Xưng hô: luôn xưng "thầy", gọi học sinh là "em". Không dùng "thầy/cô", "anh/chị", "mình", "tôi".

Tình huống hiện tại: {situation}

Nhiệm vụ: viết MỘT tin nhắn ngắn (3-6 câu), ấm áp và tự nhiên như gia sư thật đang nhắn tin.
- Phản hồi CỤ THỂ theo nội dung học sinh vừa nhắn (nếu em xưng tên, hãy gọi tên em;
  nếu em kể hoàn cảnh, hãy nhắc đến hoàn cảnh đó).
- KHÔNG giảng bài toán ở đây. KHÔNG dùng danh sách gạch đầu dòng dài.
- Có thể dùng 1-2 emoji nhẹ nhàng.
- Kết thúc bằng một câu hỏi/gợi ý mở để em bắt đầu (ví dụ: hỏi em muốn học phần nào,
  hay gợi ý làm bài chẩn đoán năng lực / lập kế hoạch ôn tập trên giao diện).
"""

    _SOCIAL_SITUATIONS = {
        "greeting": (
            "Học sinh đang chào hỏi / giới thiệu bản thân / nhờ thầy giúp học Toán chung chung. "
            "Hãy chào lại nồng nhiệt, cho em biết thầy có thể: giải thích bài, luyện quiz, "
            "lập kế hoạch ôn tập, chẩn đoán năng lực — rồi hỏi em muốn bắt đầu từ đâu."
        ),
        "motivation": (
            "Học sinh đang chán nản / lo lắng / áp lực về việc học Toán hoặc kỳ thi. "
            "Hãy đồng cảm chân thành trước (đừng sáo rỗng, đừng vội đưa giải pháp ngay câu đầu), "
            "cho em thấy cảm giác đó là bình thường và mất gốc vẫn kịp cải thiện, "
            "rồi đề xuất MỘT bước nhỏ khả thi: làm bài chẩn đoán năng lực để biết lỗ hổng ở đâu, "
            "hoặc để thầy lập kế hoạch ôn tập vừa sức."
        ),
    }

    async def social(self, state: TutorState, config: RunnableConfig) -> dict:
        """Greeting/motivation: sinh phản hồi cá nhân hóa bằng LLM mini, fallback text tĩnh."""
        intent = state.get("intent", "greeting")
        situation = self._SOCIAL_SITUATIONS.get(intent, self._SOCIAL_SITUATIONS["greeting"])
        fallback = (
            formatters.MOTIVATION_FALLBACK_TEXT
            if intent == "motivation"
            else formatters.GREETING_FALLBACK_TEXT
        )

        messages = [SystemMessage(content=self.SOCIAL_SYSTEM_PROMPT.format(situation=situation))]
        for h in (state.get("chat_history") or [])[-4:]:
            content = (h.get("content") or "")[:500]
            if h.get("role") == "user":
                messages.append(HumanMessage(content=content))
            elif h.get("role") == "assistant":
                messages.append(AIMessage(content=content))
        messages.append(HumanMessage(content=state["message"]))

        response_text = fallback
        try:
            response = await self.social_llm.ainvoke(messages)
            log_from_response(
                agent="Social", model=settings.LLM_MODEL_MINI, response=response
            )
            if response.content and str(response.content).strip():
                response_text = str(response.content).strip()
        except Exception as exc:
            logger.warning("Social node lỗi LLM, dùng fallback text: %s", exc)

        return {
            "response_text": response_text,
            "mode_used": intent,
            "result": {
                "response": response_text,
                "skill_id": None,
                "skill_name": None,
                "mastery_level": None,
                "mode_used": intent,
            },
        }

    async def plan(self, state: TutorState, config: RunnableConfig) -> dict:
        cfg = _cfg(config)
        profile = await get_mastery_profile(cfg["db"], cfg.get("user_id", 1))
        plan = await self.planner.create_plan(profile)
        text = formatters.format_plan(plan)
        return {
            "response_text": text,
            "mode_used": "plan",
            "result": {
                "response": text,
                "skill_id": None,
                "skill_name": None,
                "mastery_level": None,
                "mode_used": "plan",
            },
        }

    async def quiz(self, state: TutorState, config: RunnableConfig) -> dict:
        target_skill = state.get("skill_id") or "derivative_basic"
        skill_info = SKILLS.get(target_skill, {})
        current_mastery = state.get("current_mastery", 0.1)
        difficulty = 2 if current_mastery >= 0.5 else 1
        text = formatters.format_quiz_text(
            skill_info.get("name", target_skill), target_skill, difficulty
        )
        return {
            "response_text": text,
            "mode_used": "quiz",
            "result": {
                "response": text,
                "skill_id": target_skill,
                "skill_name": skill_info.get("name"),
                "mastery_level": round(current_mastery, 3),
                "mode_used": "quiz",
            },
        }

    async def review(self, state: TutorState, config: RunnableConfig) -> dict:
        return {
            "response_text": formatters.REVIEW_TEXT,
            "mode_used": "review",
            "result": {
                "response": formatters.REVIEW_TEXT,
                "skill_id": None,
                "skill_name": None,
                "mastery_level": None,
                "mode_used": "review",
            },
        }

    async def diagnostic(self, state: TutorState, config: RunnableConfig) -> dict:
        return {
            "response_text": formatters.DIAGNOSTIC_TEXT,
            "mode_used": "diagnostic",
            "result": {
                "response": formatters.DIAGNOSTIC_TEXT,
                "skill_id": None,
                "skill_name": None,
                "mastery_level": None,
                "mode_used": "diagnostic",
            },
        }

    # ── visualize ────────────────────────────────────────────────────────

    async def visualize(self, state: TutorState, config: RunnableConfig) -> dict:
        message = state["message"]
        chat_history = state.get("chat_history") or []
        vis_data = await self._extract_and_visualize(message, chat_history)

        skill_id = state.get("skill_id")
        skill_info = SKILLS.get(skill_id, {})
        response_text = "📊 Đồ thị đã được tạo! Xem bên dưới."
        if vis_data is None:
            response_text = (
                "⚠️ Không tìm được biểu thức toán học trong tin nhắn. "
                "Em hãy ghi rõ hàm số cần vẽ, ví dụ: *vẽ đồ thị y = 2x·sin(x) + x²·cos(x)*."
            )
        elif vis_data.get("vis_type") == "error":
            response_text = f"⚠️ {vis_data['data'].get('message', 'Không thể vẽ đồ thị.')}"
            vis_data = None

        return {
            "response_text": response_text,
            "visualization": vis_data,
            "mode_used": "visualize",
            "result": {
                "response": response_text,
                "skill_id": skill_id,
                "skill_name": skill_info.get("name"),
                "mastery_level": round(state.get("current_mastery", 0.1), 3),
                "mode_used": "visualize",
                "visualization": vis_data,
            },
        }

    EXPR_EXTRACTOR_PROMPT = """You are a math expression extractor for a Python/SymPy plotter.

Given a student's message and recent conversation history, extract the single mathematical
function f(x) that should be plotted. Return ONLY a valid Python expression using x as the
variable — nothing else.

Rules:
- Use ** for exponentiation (e.g. x**2, not x^2)
- Use sin(x), cos(x), tan(x), log(x), exp(x), sqrt(x)
- No integration constants (remove C, K, etc.)
- No equals sign, no "y =", just the right-hand side
- If the student said "nó" / "cái đó" / "đáp án đó", look for the expression in history
- If there is truly no plottable expression, reply with exactly: NONE

Examples of valid replies:
  x**2*sin(x)
  2*x*sin(x) + x**2*cos(x)
  exp(-x)*cos(x)
  NONE
"""

    async def _extract_and_visualize(
        self, message: str, chat_history: list[dict]
    ) -> dict | None:
        expr = await self._extract_plottable_expr_llm(message, chat_history)
        if not expr:
            return None
        return self.visualizer.generate_function_plot(expr)

    async def _extract_plottable_expr_llm(
        self, message: str, chat_history: list[dict]
    ) -> str | None:
        from app.utils.cost_tracker import log_call

        history_snippet = ""
        if chat_history:
            lines = []
            for h in chat_history[-4:]:
                role = "Học sinh" if h["role"] == "user" else "Gia sư"
                lines.append(f"{role}: {h['content'][:300]}")
            history_snippet = "\n".join(lines)

        user_content = f"Tin nhắn học sinh: {message}"
        if history_snippet:
            user_content += f"\n\nLịch sử hội thoại gần đây:\n{history_snippet}"

        try:
            response = await self.openai_client.responses.create(
                model=settings.LLM_MODEL_NANO,
                input=[
                    {"role": "system", "content": self.EXPR_EXTRACTOR_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            )
            expr = response.output_text.strip()

            input_tokens = getattr(response.usage, "input_tokens", 0) if response.usage else 0
            output_tokens = getattr(response.usage, "output_tokens", 0) if response.usage else 0
            log_call(agent="ExprExtractor", model=settings.LLM_MODEL_NANO,
                     input_tokens=input_tokens, output_tokens=output_tokens)

            if not expr or expr.upper() == "NONE":
                return None
            if "x" not in expr.lower():
                return None
            return expr
        except Exception:
            return None

    # ── teach (explain / answer / abstain handoff) ───────────────────────

    async def teach(
        self, state: TutorState, config: RunnableConfig, writer: StreamWriter = None
    ) -> dict:
        cfg = _cfg(config)
        db = cfg["db"]
        user_id = cfg.get("user_id", 1)
        message = state["message"]
        intent = state.get("intent", "explain")
        skill_id = state.get("skill_id")
        skill_ids = state.get("skill_ids") or []
        formula_ids = state.get("formula_ids") or []
        chat_history = state.get("chat_history") or []
        current_mastery = state.get("current_mastery", 0.1)

        # intent="answer" ("cho em đáp án luôn") chỉ được đổi mode khi học sinh
        # để chế độ "auto". Em đã tự chọn Socratic/Exam thì lựa chọn đó thắng —
        # không thể xin đáp án để lách chính chế độ mình vừa bật.
        mode = state.get("mode", "socratic")
        if intent == "answer" and state.get("requested_mode", "auto") == "auto":
            mode = "answer"

        # F-01: học sinh bí lặp lại cùng một bước → ép worked micro-step.
        # Chỉ áp cho socratic (answer/exam vốn đã đưa lời giải đầy đủ).
        extra_directive = None
        if mode == "socratic":
            stuck_count = count_consecutive_stuck(chat_history, message)
            extra_directive = build_stuck_directive(stuck_count)
            if extra_directive:
                logger.info("F-01: học sinh bí %d lần → ép worked micro-step", stuck_count)

        prereq_gaps = []
        if skill_id:
            prereq_gaps = await identify_gaps(db, skill_id, user_id)

        # Nhánh explain mặc định giữ check từ khóa vẽ đồ thị (port từ else-branch cũ)
        vis_data = None
        if mode != "answer":
            graph_keywords = ["đồ thị", "vẽ", "biểu đồ", "minh họa", "hình ảnh"]
            if any(kw in message.lower() for kw in graph_keywords):
                vis_data = await self._extract_and_visualize(message, chat_history)
                if vis_data and vis_data.get("vis_type") == "error":
                    vis_data = None

        writer = _writer_if_streaming(config, writer)
        streamed = False
        if writer is not None:
            # Stream mode: giống handle_message_stream cũ — không reflection
            full_response = ""
            async for token in self.teacher.respond_stream(
                question=message,
                mode=mode,
                mastery_level=state.get("mastery_level", "beginner"),
                prerequisite_gaps=prereq_gaps,
                chat_history=chat_history,
                skill_id=skill_id,
                skill_ids=skill_ids,
                formula_ids=formula_ids,
                masteries=state.get("masteries") or {},
                p_mastery=current_mastery,
                extra_directive=extra_directive,
            ):
                full_response += token
                writer(json.dumps({"type": "token", "content": token}, ensure_ascii=False))
            response_text = full_response
            streamed = True
        else:
            response_text = await self.teacher.respond(
                question=message,
                mode=mode,
                mastery_level=state.get("mastery_level", "beginner"),
                prerequisite_gaps=prereq_gaps,
                chat_history=chat_history,
                skill_id=skill_id,
                skill_ids=skill_ids,
                formula_ids=formula_ids,
                masteries=state.get("masteries") or {},
                p_mastery=current_mastery,
                trace_id=cfg.get("trace_id"),
                extra_directive=extra_directive,
            )

        # Thẻ <answer> chỉ dành cho evaluation pipeline — không lộ ra chat.
        # Socratic: bỏ luôn nội dung (đáp án cuối phải giữ kín).
        response_text = strip_answer_tag_for_chat(
            response_text, reveal=(mode != "socratic")
        )

        skill_info = SKILLS.get(skill_id, {})
        result = {
            "response": response_text,
            "skill_id": skill_id,
            "skill_ids": skill_ids,
            "formula_ids": formula_ids,
            "skill_name": skill_info.get("name"),
            "mastery_level": round(current_mastery, 3),
            "mode_used": mode,
        }
        if mode != "answer":
            result["visualization"] = vis_data

        return {
            "response_text": response_text,
            "visualization": vis_data,
            "mode_used": mode,
            "streamed_tokens": streamed,
            "result": result,
        }

    # ── grading pipeline ─────────────────────────────────────────────────

    async def grade_extract(self, state: TutorState, config: RunnableConfig) -> dict:
        message = state["message"]
        chat_history = state.get("chat_history") or []
        snapshot = state.get("session_state") or SessionSnapshot()

        original_problem = snapshot.current_problem or find_original_question(
            chat_history, message
        )
        extraction = await self.extractor.extract(
            message=message,
            original_problem=original_problem,
            chat_history=chat_history,
            # Socratic + bài đang treo chờ em trả lời → nhiều khả năng tin nhắn
            # là bước trung gian chứ không phải bài nộp.
            in_socratic_dialogue=(
                state.get("mode") == "socratic" and snapshot.awaiting_answer
            ),
        )
        return {"extraction": extraction, "abstained": not extraction.gradable}

    async def grade_verify(self, state: TutorState, config: RunnableConfig) -> dict:
        extraction = state.get("extraction") or GradingExtraction()
        return {"verification": verify(extraction)}

    GRADE_FEEDBACK_PROMPT = """Bạn là "thầy" — giám khảo chấm bài Toán 12.
Trong feedback và lời giải: xưng "thầy", gọi học sinh là "em". Không dùng "thầy/cô", "anh/chị", "mình", "tôi".

Bài toán yêu cầu các kỹ năng: {skill_names_list}

Công thức áp dụng:
{formulas_list}
{cas_block}{socratic_block}
QUY TRÌNH CHẤM (BẮT BUỘC theo thứ tự):
1. Tự giải đề bài để có đáp án chuẩn.
2. Đối chiếu TỪNG KẾT LUẬN trong bài làm của học sinh với đáp án chuẩn.
   Đặc biệt chú ý lỗi ĐẢO NGƯỢC: cực đại/cực tiểu, đồng biến/nghịch biến,
   lớn nhất/nhỏ nhất, dấu bất đẳng thức.
3. Kiểm tra học sinh áp dụng đúng công thức và điều kiện xác định.
4. Với từng kỹ năng, xác định học sinh làm đúng hay sai ở bước nào.
5. Cho điểm 0.0-1.0 và confidence (độ chắc chắn của chính bạn về verdict).

RÀNG BUỘC NHẤT QUÁN:
- is_correct = true CHỈ KHI mọi kết luận của học sinh đều đúng.
- feedback, score, is_correct, skills_assessed PHẢI thống nhất với nhau:
  không được vừa khen đúng vừa chỉ ra lỗi sai.
- correct_solution: lời giải đúng đầy đủ bằng tiếng Việt.
"""

    CAS_VERDICT_BLOCK = """
KẾT QUẢ KIỂM CHỨNG TỰ ĐỘNG (SymPy — độ tin cậy tuyệt đối):
Đáp án của học sinh được máy xác nhận là: {cas_verdict}.
Nhận xét của bạn BẮT BUỘC phải nhất quán với kết luận này.
({cas_details})
"""

    # Học sinh đang ở chế độ Socratic: feedback là kênh duy nhất còn hiển thị khi
    # bài làm sai (khối "Lời giải đúng" đã bị ẩn), nên chính nó không được lộ đáp án.
    SOCRATIC_FEEDBACK_BLOCK = """
CHẾ ĐỘ SOCRATIC — RÀNG BUỘC BỔ SUNG CHO feedback (ưu tiên cao):
Học sinh đang tự tìm lời giải. Nếu bài làm SAI, trường "feedback" TUYỆT ĐỐI KHÔNG
được chứa đáp án đúng (không nêu con số, biểu thức kết quả, hay giá trị đúng của đề).
Chỉ được nói em sai ở BƯỚC NÀO và sai vì lý do gì, để em tự tính lại.
Vẫn điền "correct_solution" đầy đủ như bình thường — hệ thống tự quyết định có hiện hay không.
"""

    async def grade_feedback(self, state: TutorState, config: RunnableConfig) -> dict:
        extraction = state.get("extraction") or GradingExtraction()
        verification = state.get("verification")
        message = state["message"]
        skill_ids = normalize_skill_ids(state.get("skill_ids"))
        chat_history = state.get("chat_history") or []

        cas_block = ""
        if verification is not None and verification.verified:
            cas_block = self.CAS_VERDICT_BLOCK.format(
                cas_verdict="ĐÚNG" if verification.verdict == "correct" else "SAI",
                cas_details=verification.details,
            )

        system_prompt = self.GRADE_FEEDBACK_PROMPT.format(
            skill_names_list=format_skill_names(skill_ids),
            formulas_list=format_formulas(state.get("formula_ids")),
            cas_block=cas_block,
            socratic_block=(
                self.SOCRATIC_FEEDBACK_BLOCK if state.get("mode") == "socratic" else ""
            ),
        )

        question = extraction.problem_statement or find_original_question(
            chat_history, message
        )
        history_block = AssessorAgent._format_history(chat_history)
        user_prompt = (
            f"CÂU HỎI:\n{question}\n{history_block}"
            f"CÂU TRẢ LỜI CỦA HỌC SINH (cần chấm):\n{message}\n\n"
            "Hãy chấm điểm và phân tích."
        )

        assessment = PedagogyAssessment(
            is_correct=False,
            score=0.0,
            confidence=0.0,
            error_type="unknown",
            feedback="Không thể phân tích lời giải.",
            correct_solution="Không thể phân tích lời giải.",
        )
        try:
            output = await self.feedback_llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
            raw = output.get("raw")
            if raw is not None:
                log_from_response(
                    agent="Assessor", model=settings.LLM_MODEL_MINI, response=raw
                )
            if output.get("parsed") is not None:
                assessment = output["parsed"]
        except Exception as exc:
            logger.warning("grade_feedback lỗi, dùng fallback assessment: %s", exc)

        # KHÔNG reconcile với CAS ở đây — grade_policy cần assessment thô
        # để phát hiện mismatch LLM/CAS trước, rồi mới reconcile để hiển thị.
        assessment = self._enforce_internal_consistency(assessment)
        if not assessment.overall_feedback:
            assessment.overall_feedback = assessment.feedback
        return {"assessment": assessment}

    @staticmethod
    def _enforce_internal_consistency(assessment: PedagogyAssessment) -> PedagogyAssessment:
        """Port _enforce_consistency cũ: skill fail nào đó → is_correct=False, clamp score."""
        skill_results = [v.passed for v in assessment.skills_assessed.values()]
        updated = assessment.model_copy()
        if skill_results and not all(skill_results):
            updated.is_correct = False
        if not updated.is_correct and updated.score >= 1.0:
            partial = (
                round(sum(skill_results) / len(skill_results), 2)
                if skill_results
                else 0.5
            )
            updated.score = min(partial, 0.9)
        if not updated.is_correct and updated.error_type == "none":
            updated.error_type = "conceptual"
        return updated

    async def grade_policy(self, state: TutorState, config: RunnableConfig) -> dict:
        cfg = _cfg(config)
        db = cfg["db"]
        user_id = cfg.get("user_id", 1)
        skill_id = state.get("skill_id")
        skill_ids = state.get("skill_ids") or []
        current_mastery = state.get("current_mastery", 0.1)

        extraction = state.get("extraction")
        verification = state.get("verification")
        assessment = state.get("assessment")

        decision = decide(
            extraction,
            verification,
            assessment,
            skill_ids,
            settings.GRADING_CONFIDENCE_THRESHOLD,
        )

        new_mastery = current_mastery
        if decision.should_write:
            updates = await update_mastery(
                db,
                {sid: {"passed": passed} for sid, passed in decision.skills_assessed.items()},
                user_id=user_id,
            )
            if isinstance(updates, dict):
                new_mastery = updates.get(skill_id, current_mastery)
            else:
                new_mastery = updates

        if decision.mismatch:
            from app.utils.langfuse_client import score_trace

            trace_id = cfg.get("trace_id")
            if trace_id:
                score_trace(
                    trace_id,
                    name="grading_verdict_mismatch",
                    value=1.0,
                    comment=(
                        f"LLM={assessment.is_correct if assessment else None} "
                        f"CAS={verification.verdict if verification else None} :: "
                        f"{extraction.candidate_expr if extraction else None}"
                    ),
                )
            logger.warning(
                "Grading mismatch: LLM=%s CAS=%s (candidate=%s)",
                assessment.is_correct if assessment else None,
                verification.verdict if verification else None,
                extraction.candidate_expr if extraction else None,
            )

        hedged = (
            decision.reason == "unverified_low_confidence"
            and assessment is not None
        )
        display_assessment = reconcile_assessment(assessment, verification)
        # Socratic + em làm SAI → chấm và chỉ chỗ vướng, nhưng giữ kín lời giải
        # để em còn cơ hội tự sửa. Em làm ĐÚNG thì hiện bình thường (em xong bài rồi).
        reveal_solution = not (
            state.get("mode") == "socratic" and not display_assessment.is_correct
        )
        response_text = formatters.format_assessment(
            display_assessment, hedged=hedged, reveal_solution=reveal_solution
        )

        skill_info = SKILLS.get(skill_id, {})
        return {
            "mastery_decision": decision,
            "assessment": display_assessment,
            "new_mastery": new_mastery,
            "response_text": response_text,
            "mode_used": "assess",
            "result": {
                "response": response_text,
                "skill_id": skill_id,
                "skill_ids": skill_ids,
                "formula_ids": state.get("formula_ids") or [],
                "skill_name": skill_info.get("name"),
                "mastery_level": round(new_mastery, 3),
                "mode_used": "assess",
            },
        }

    # ── finalize ─────────────────────────────────────────────────────────

    async def finalize(
        self, state: TutorState, config: RunnableConfig, writer: StreamWriter = None
    ) -> dict:
        cfg = _cfg(config)
        db = cfg["db"]
        session_id = cfg.get("session_id")
        result = state.get("result") or {
            "response": state.get("response_text", ""),
            "skill_id": state.get("skill_id"),
            "mode_used": state.get("mode_used", state.get("mode", "socratic")),
        }

        # ── Ghi SessionState tường minh ────────────────────────────────
        if session_id is not None:
            extraction = state.get("extraction")
            classification = state.get("classification")
            try:
                if extraction is not None and extraction.gradable:
                    await upsert_session_state(
                        db,
                        session_id,
                        current_problem=extraction.problem_statement,
                        awaiting_answer=False,
                        last_skill_ids=state.get("skill_ids") or [],
                    )
                elif (
                    classification is not None
                    and classification.contains_problem
                    and state.get("intent") in ("explain", "answer", "assess")
                ):
                    await upsert_session_state(
                        db,
                        session_id,
                        current_problem=state["message"],
                        awaiting_answer=state.get("mode") == "socratic",
                        last_skill_ids=state.get("skill_ids") or [],
                    )
            except Exception as exc:
                # Session state là tối ưu hóa routing — không được làm hỏng response
                logger.warning("Không ghi được SessionState: %s", exc)

        # ── SSE frames ─────────────────────────────────────────────────
        writer = _writer_if_streaming(config, writer)
        if writer is not None:
            full = result.get("response", "")
            if state.get("streamed_tokens"):
                # teach đã stream token → done frame gọn (giữ bất đối xứng cũ)
                writer(json.dumps(
                    {"type": "done", "full_response": full}, ensure_ascii=False
                ))
            else:
                writer(json.dumps({"type": "token", "content": full}, ensure_ascii=False))
                writer(json.dumps({
                    "type": "done",
                    "full_response": full,
                    "skill_id": result.get("skill_id"),
                    "skill_ids": result.get("skill_ids") or [],
                    "formula_ids": result.get("formula_ids") or [],
                    "skill_name": result.get("skill_name"),
                    "mastery_level": result.get("mastery_level"),
                    "mode_used": result.get("mode_used", state.get("mode", "socratic")),
                    "visualization": result.get("visualization"),
                }, ensure_ascii=False))

        return {"result": result}
