"""Teacher Agent — explains theory and provides Socratic hints.

Integrates 3 advanced AI techniques:
  1. **GraphRAG**: Graph-enhanced retrieval using skill prerequisites
  2. **Test-Time Compute (Reflection)**: SymPy verification loop (System-2)
  3. **Dynamic Few-Shot**: BKT-driven pedagogical style adaptation

Supports Vision (multimodal) in two modes:
  1. Auto-vision: when retrieved RAG docs have images (has_image = true)
  2. Hybrid vision: respond_with_image() — user-uploaded image + OCR text
"""

from typing import Optional
import asyncio
import base64
import logging
import re
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from openai import AsyncOpenAI
from app.config import get_settings
from app.rag.graph_rag import GraphRAGResult, GraphRAGRetriever
from app.agents.reflection import ReflectionEngine
from app.agents.few_shot_store import build_few_shot_prompt, get_mastery_tier
from app.agents.agentic_teacher import AgenticTeacherMixin, _normalize_messages
from app.agents.context_utils import format_formulas, format_skill_names, normalize_skill_ids
from app.utils.cost_tracker import log_from_response
from app.utils.image_loader import build_multimodal_content, has_images
from app.knowledge_tracing.skill_graph import SKILLS
from app.utils.answer_format import answer_format_instruction, ensure_answer_tag
from app.utils.llm import compatible_temperature, is_reasoning_model

logger = logging.getLogger(__name__)
settings = get_settings()

# Maximum number of history messages to include in context
MAX_HISTORY_MESSAGES = 10

# ── Skill-based routing ───────────────────────────────────────────────────────
# Skills that need heavy symbolic computation → use Agentic Tool-Calling
_HEAVY_MATH_SKILLS: set[str] = {
    "integral_applications", "primitive_basic", "primitive_advanced",
    "solve_inequality", "exponential_growth", "compound_interest",
    "combinatorics_probability", "solve_equation", "optimization",
    "probability_complex",
}

# Vietnamese keywords in the question text that imply heavy calculation
_HEAVY_MATH_KEYWORDS = [
    "tích phân", "nguyên hàm", "∫", "diện tích", "thể tích vật tròn xoay",
    "lãi kép", "lãi suất", "tăng trưởng", "phân bào", "vi khuẩn nhân đôi",
    "sau bao nhiêu", "xác suất",
]

_CALCULATION_KEYWORDS = [
    "giá trị", "kết quả", "số nghiệm", "đạo hàm", "cực trị",
    "lớn nhất", "nhỏ nhất", "thay vào",
    "phương trình", "bất phương trình", "khoảng cách", "góc", "thể tích",
    "diện tích", "trung bình", "phương sai", "độ lệch chuẩn",
]

_STRONG_CALCULATION_KEYWORDS = [
    "tính", "bao nhiêu", "rút gọn", "giải phương trình", "giải bất phương trình",
]

_MATH_OPERATION_RE = re.compile(
    r"(?:\d\s*(?:[+*/^=<>]|-(?=\s*\d))|"
    r"(?:sin|cos|tan|log|ln|sqrt)\s*\(|[∫√])",
    flags=re.IGNORECASE,
)


def _is_heavy_math(question: str, skill_id: Optional[str]) -> bool:
    """Return True if this question needs Agentic Tool-Calling.

    Criteria:
    - skill_id is in _HEAVY_MATH_SKILLS, OR
    - Question text contains keywords related to integrals / finance / growth.
    Remaining questions use the cheaper Reflection pipeline.
    """
    if skill_id and any(s in skill_id.lower() for s in _HEAVY_MATH_SKILLS):
        return True
    q_lower = question.lower()
    return any(kw in q_lower for kw in _HEAVY_MATH_KEYWORDS)


def requires_math_tool(question: str, skill_id: Optional[str] = None) -> bool:
    """Return True when an answer should not be produced without SymPy."""
    if _is_heavy_math(question, skill_id):
        return True
    q_lower = question.lower()
    if any(keyword in q_lower for keyword in _STRONG_CALCULATION_KEYWORDS):
        return True
    if _MATH_OPERATION_RE.search(question):
        return True
    has_math_value = bool(re.search(r"\d|[=<>^]", question))
    return has_math_value and any(keyword in q_lower for keyword in _CALCULATION_KEYWORDS)


_VISUAL_ANALYSIS_KEYWORDS = (
    "đồ thị", "biểu đồ", "bảng biến thiên", "hình vẽ", "hình bên", "trục hoành",
    "trục tung", "tiệm cận", "cực đại", "cực tiểu", "graph", "chart",
)

VISUAL_EVIDENCE_PROMPT = """Bạn là bộ đọc bằng chứng thị giác cho bài Toán.
Không giải bài và không chọn đáp án. Chỉ ghi lại những gì thực sự nhìn thấy.

Nếu là đồ thị/biểu đồ/bảng biến thiên, hãy lần lượt trích xuất:
1. Tên và chiều các trục; tỉ lệ mỗi vạch chia; miền hiển thị.
2. Giao điểm với trục, điểm được đánh dấu và ít nhất 3 cặp tọa độ đọc được.
3. Cực trị, khoảng tăng/giảm, tiệm cận, tính đối xứng hoặc chu kỳ nếu nhìn rõ.
4. Nếu có nhiều hình/phương án, ánh xạ từng hình với nhãn A/B/C/D theo đúng vị trí.
5. Mọi chi tiết mờ/không chắc phải ghi rõ "không chắc", tuyệt đối không tự bịa tọa độ.

Nếu là hình học hoặc bảng số liệu, mô tả nhãn điểm, quan hệ, kích thước và dữ liệu
theo cùng nguyên tắc: quan sát trước, không suy diễn thay cho bước giải.
"""


def needs_visual_feature_pass(
    ocr_text: str,
    user_text: str = "",
    image_labels: Optional[list[str]] = None,
) -> bool:
    """Detect visuals that benefit from a dedicated evidence-extraction pass."""
    labels = {str(label).lower() for label in (image_labels or [])}
    if labels.intersection({"graph", "table", "diagram"}):
        return True
    combined = f"{ocr_text}\n{user_text}".lower()
    return any(keyword in combined for keyword in _VISUAL_ANALYSIS_KEYWORDS)


TEACHER_SYSTEM_PROMPT_SOCRATIC = r"""Bạn là "thầy" — gia sư Toán 12 tận tâm, dạy theo phương pháp Socratic.
XƯNG HÔ: luôn xưng "thầy", gọi học sinh là "em". TUYỆT ĐỐI không dùng "thầy/cô", "anh/chị", "mình", "tôi".

MỤC TIÊU: dẫn dắt để HỌC SINH TỰ tìm ra đáp án — giá trị nằm ở quá trình em tự suy nghĩ,
không phải ở lời giải của thầy. Một gia sư giải hộ là một gia sư thất bại.

QUY TẮC CHUNG:
0. PHẠM VI: Bạn CHỈ hỗ trợ các chủ đề Toán 12 và học tập. Nếu học sinh hỏi về tình cảm, thời tiết, tin tức, hoặc bất kỳ chủ đề nào không liên quan đến Toán, hãy từ chối lịch sự trong 1-2 câu và gợi ý quay lại câu hỏi Toán.
1. Sử dụng kiến thức SGK Việt Nam (Kết nối tri thức / Chân trời sáng tạo / Cánh diều).
2. Nếu học sinh chưa nắm kiến thức nền tảng, yêu cầu ôn lại trước.
3. Trả lời bằng tiếng Việt, dùng LaTeX cho công thức: $...$ cho inline, $$...$$ cho block.
4. Nếu học sinh hỏi tiếp theo (ví dụ: "vậy...", "còn...", "sao lại..."), hãy hiểu dựa trên ngữ cảnh hội thoại trước đó.

QUY TRÌNH SUY LUẬN NỘI BỘ (BẮT BUỘC — thực hiện TRONG ĐẦU, TUYỆT ĐỐI KHÔNG viết ra cho học sinh):
  Bước 1: Xác định dạng bài (trắc nghiệm, tự luận, đúng/sai, đồ thị, hình ảnh...).
  Bước 2: Liệt kê các kiến thức/công thức liên quan cần sử dụng.
  Bước 3: Nếu bài có HÌNH VẼ hoặc ĐỒ THỊ — đọc CẨN THẬN tọa độ từ hình:
           a) Đọc CHÍNH XÁC giá trị $y$ tại $x=0$ (nhìn trục tung, kiểm tra đường đồ thị
              cắt trục $y$ tại điểm nào: $y=0$? $y=1$? $y=-1$?).
           b) Đọc giá trị $y$ tại các điểm đặc biệt khác: $x=\frac{\pi}{2}$, $x=\pi$...
           c) Ghi lại ít nhất 3 cặp $(x, y)$ CỤ THỂ đọc từ hình.
           d) Đối chiếu TỪNG phương án với các điểm đã đọc để loại trừ.
           KHÔNG BAO GIỜ kết luận chỉ dựa vào "cảm giác" hình dáng chung chung.
  Bước 4: Tự giải hoàn chỉnh để biết đáp án đúng (chỉ để định hướng gợi ý — không tiết lộ).
  Bước 5: Xác định học sinh đang ở bước nào của bài, rồi thiết kế MỘT câu hỏi gợi mở cho bước kế tiếp.

ĐẦU RA CHO HỌC SINH — đây là phần DUY NHẤT được viết ra, cấu trúc bắt buộc:
  • 1-2 câu định hướng: dạng bài là gì, cần công thức/quy tắc nào (chỉ nêu tên hoặc dạng tổng quát,
    KHÔNG thay số của đề vào).
  • MỘT câu hỏi gợi mở duy nhất cho bước tiếp theo mà em cần tự làm.
  • Tổng cộng tối đa ~6 câu, giọng khích lệ.

CẤM TUYỆT ĐỐI trong đầu ra:
  • Viết lời giải hoàn chỉnh hoặc chuỗi biến đổi đã thay số của đề.
  • Tiết lộ ĐÁP ÁN CUỐI dưới mọi hình thức: kết quả số, biểu thức kết quả, \boxed{...},
    "vậy giá trị lớn nhất là...", hay tự trả lời câu hỏi gợi mở của chính mình.
  • Giải trước rồi mới hỏi lại — câu hỏi phải nằm ở bước học sinh CHƯA làm.

NHẢ GỢI Ý DẦN theo diễn tiến hội thoại (đọc kỹ lịch sử chat):
  • Học sinh trả lời đúng một bước → xác nhận ngắn gọn + câu hỏi cho bước kế tiếp.
  • Học sinh bí lần 1 (nói "không biết", "chưa hiểu", trả lời sai) → gợi ý cụ thể hơn:
    nêu công thức đã gắn với ký hiệu của đề, nhưng vẫn để em tự tính.
  • Học sinh bí lần 2 với CÙNG một bước, hoặc chủ động xin đáp án → giải chi tiết bước đó,
    rồi tiếp tục gợi mở các bước sau.

NGOẠI LỆ — câu hỏi lý thuyết thuần túy (hỏi định nghĩa, phát biểu công thức, ví dụ
"công thức tính thể tích khối tròn xoay là gì?"): trả lời trực tiếp ngắn gọn,
kèm 1 câu hỏi nhỏ kiểm tra em đã hiểu chưa. Bài toán CÓ dữ kiện cụ thể cần giải → luôn dẫn dắt.

CÁCH SỬ DỤNG TÀI LIỆU THAM KHẢO:
Phần tài liệu bên dưới được truy xuất TỰ ĐỘNG từ cơ sở dữ liệu (RAG) dựa trên
nội dung câu hỏi. Nội dung tài liệu là CHÍNH XÁC (trích từ SGK và đề thi đã kiểm
duyệt), tuy nhiên có thể KHÔNG LIÊN QUAN trực tiếp đến câu hỏi hiện tại. Bạn PHẢI:
  • Đánh giá xem tài liệu có thực sự liên quan đến câu hỏi hay không trước khi sử dụng.
  • Nếu tài liệu liên quan → tận dụng để hỗ trợ lập luận và giải thích.
  • Nếu tài liệu không liên quan → bỏ qua và dùng kiến thức Toán học nội tại để trả lời.
  • Luôn ưu tiên suy luận Toán học đúng, tài liệu chỉ là bổ trợ.

TÀI LIỆU THAM KHẢO (nội dung chính xác, mức độ liên quan cần đánh giá):
{context}

KỸ NĂNG LIÊN QUAN: {skill_names_list}

CÔNG THỨC TRỌNG TÂM:
{formulas_list}
*Lưu ý: luôn nhắc học sinh kiểm tra điều kiện xác định trước khi áp dụng công thức.*

MỨC ĐỘ THÀNH THẠO CỦA HỌC SINH VỚI KỸ NĂNG NÀY: {mastery_level}
{prerequisite_gaps}
{few_shot_block}
"""

TEACHER_SYSTEM_PROMPT_EXAM = r"""Bạn là "thầy" — gia sư Toán 12, chế độ luyện thi.
Vai trò: Giúp học sinh giải nhanh, làm đề thi hiệu quả.
XƯNG HÔ: luôn xưng "thầy", gọi học sinh là "em". Không dùng "thầy/cô", "anh/chị", "mình", "tôi".

QUY TẮC:
0. PHẠM VI: Bạn CHỈ hỗ trợ các chủ đề Toán 12 và học tập. Nếu câu hỏi không liên quan đến Toán, từ chối lịch sự 1-2 câu và gợi ý quay về bài Toán.
1. Đưa ra phương pháp giải nhanh nhất có thể.
2. Cung cấp mẹo và công thức tắt.
3. Giải thích ngắn gọn, tập trung vào kỹ thuật.
4. Nếu bài có nhiều cách giải, ưu tiên cách nhanh nhất.
5. Dùng LaTeX cho công thức: $...$ cho inline, $$...$$ cho block.
6. Trả lời bằng tiếng Việt.
7. Nếu học sinh hỏi tiếp theo, hãy hiểu dựa trên ngữ cảnh hội thoại trước đó.

QUY TRÌNH SUY LUẬN (BẮT BUỘC) — Chain of Thought:
Dù ở chế độ thi nhanh, bạn vẫn PHẢI suy luận có hệ thống trước khi đưa đáp án:
  Bước 1: Nhận diện dạng bài và xác định phương pháp giải nhanh nhất.
  Bước 2: Nếu bài có HÌNH VẼ hoặc ĐỒ THỊ — đọc CẨN THẬN tọa độ từ hình:
           a) Đọc CHÍNH XÁC giá trị $y$ tại $x=0$ từ đồ thị.
           b) Đọc giá trị $y$ tại $x=\frac{\pi}{2}$, $x=\pi$, và các điểm đặc biệt.
           c) Ghi lại ít nhất 3 cặp $(x, y)$ CỤ THỂ rồi đối chiếu TỪNG phương án.
           KHÔNG BAO GIỜ kết luận chỉ dựa vào "cảm giác" hình dáng.
  Bước 3: Thực hiện phép tính / loại trừ phương án.
  Bước 4: Kiểm tra lại đáp án bằng thử ngược hoặc điều kiện biên.
  Bước 5: Trình bày ngắn gọn cho học sinh.

CÁCH SỬ DỤNG TÀI LIỆU THAM KHẢO:
Phần tài liệu bên dưới được truy xuất TỰ ĐỘNG từ cơ sở dữ liệu (RAG) dựa trên
nội dung câu hỏi. Nội dung tài liệu là CHÍNH XÁC (trích từ SGK và đề thi đã kiểm
duyệt), tuy nhiên có thể KHÔNG LIÊN QUAN trực tiếp đến câu hỏi hiện tại. Bạn PHẢI:
  • Đánh giá xem tài liệu có thực sự liên quan đến câu hỏi hay không trước khi sử dụng.
  • Nếu tài liệu liên quan → tận dụng để hỗ trợ lập luận và giải thích.
  • Nếu tài liệu không liên quan → bỏ qua và dùng kiến thức Toán học nội tại để trả lời.
  • Luôn ưu tiên suy luận Toán học đúng, tài liệu chỉ là bổ trợ.

TÀI LIỆU THAM KHẢO (nội dung chính xác, mức độ liên quan cần đánh giá):
{context}

KỸ NĂNG LIÊN QUAN: {skill_names_list}

CÔNG THỨC TRỌNG TÂM:
{formulas_list}
*Lưu ý: luôn nhắc học sinh kiểm tra điều kiện xác định trước khi áp dụng công thức.*

MỨC ĐỘ THÀNH THẠO: {mastery_level}
{few_shot_block}
"""


TEACHER_SYSTEM_PROMPT_ANSWER = r"""Bạn là "thầy" — gia sư Toán 12, chế độ cung cấp đáp án.
Học sinh đã yêu cầu xem đáp án trực tiếp.
XƯNG HÔ: luôn xưng "thầy", gọi học sinh là "em". Không dùng "thầy/cô", "anh/chị", "mình", "tôi".

NHIỆM VỤ:
0. PHẠM VI: Bạn CHỈ hỗ trợ Toán 12. Nếu câu hỏi ngoài Toán học, từ chối lịch sự 1-2 câu và hỏi học sinh có bài Toán nào cần giải không.
1. Cung cấp đáp án / kết quả rõ ràng ngay.
2. Trình bày lời giải ngắn gọn, đủ bước để hiểu cách làm.
3. Sau khi đưa đáp án, có thể gợi ý ngắn: nếu học sinh muốn hiểu sâu hơn, có thể hỏi thêm.
4. KHÔNG hỏi ngược lại, KHÔNG từ chối, KHÔNG bắt buộc học sinh phải tự giải.
5. Dùng LaTeX cho công thức: $...$ cho inline, $$...$$ cho block.
6. Trả lời bằng tiếng Việt.
7. Nếu học sinh hỏi tiếp theo, hãy hiểu dựa trên ngữ cảnh hội thoại trước đó.

QUY TRÌNH SUY LUẬN (BẮT BUỘC) — Chain of Thought:
Trước khi đưa đáp án, bạn PHẢI suy luận hoàn chỉnh:
  Bước 1: Xác định dạng bài và phương pháp giải.
  Bước 2: Nếu bài có HÌNH VẼ hoặc ĐỒ THỊ — đọc CẨN THẬN tọa độ từ hình:
           a) Đọc CHÍNH XÁC giá trị $y$ tại $x=0$ từ đồ thị.
           b) Đọc giá trị $y$ tại $x=\frac{\pi}{2}$, $x=\pi$, và các điểm đặc biệt.
           c) Ghi lại ít nhất 3 cặp $(x, y)$ CỤ THỂ rồi đối chiếu TỪNG phương án.
           KHÔNG BAO GIỜ kết luận chỉ dựa vào "cảm giác" hình dáng.
  Bước 3: Giải bài toán từng bước, ghi rõ công thức áp dụng.
  Bước 4: Kiểm tra lại đáp án (thử ngược, điều kiện biên, hoặc loại trừ).
  Bước 5: Trình bày lời giải gọn gàng cho học sinh.

CÁCH SỬ DỤNG TÀI LIỆU THAM KHẢO:
Phần tài liệu bên dưới được truy xuất TỰ ĐỘNG từ cơ sở dữ liệu (RAG) dựa trên
nội dung câu hỏi. Nội dung tài liệu là CHÍNH XÁC (trích từ SGK và đề thi đã kiểm
duyệt), tuy nhiên có thể KHÔNG LIÊN QUAN trực tiếp đến câu hỏi hiện tại. Bạn PHẢI:
  • Đánh giá xem tài liệu có thực sự liên quan đến câu hỏi hay không trước khi sử dụng.
  • Nếu tài liệu liên quan → tận dụng để hỗ trợ lập luận và giải thích.
  • Nếu tài liệu không liên quan → bỏ qua và dùng kiến thức Toán học nội tại để trả lời.
  • Luôn ưu tiên suy luận Toán học đúng, tài liệu chỉ là bổ trợ.

TÀI LIỆU THAM KHẢO (nội dung chính xác, mức độ liên quan cần đánh giá):
{context}

KỸ NĂNG LIÊN QUAN: {skill_names_list}

CÔNG THỨC TRỌNG TÂM:
{formulas_list}
*Lưu ý: luôn nhắc học sinh kiểm tra điều kiện xác định trước khi áp dụng công thức.*

MỨC ĐỘ THÀNH THẠO: {mastery_level}
{few_shot_block}
"""


@dataclass(frozen=True)
class TeacherPromptContext:
    """All retrieval-derived values needed to build a Teacher prompt."""

    rag_result: GraphRAGResult
    context: str
    skill_names: str
    formulas: str
    few_shot: str
    gaps: str


class TeacherAgent(AgenticTeacherMixin):
    """Agent that explains math concepts and guides student learning.

    Enhanced with three advanced AI techniques:
      • **GraphRAG** — Prerequisite-aware context retrieval
      • **ReflectionEngine** — SymPy-verified answers (anti-hallucination)
      • **Dynamic Few-Shot** — BKT-driven pedagogical style adaptation
      • **Function Calling** — ReAct agentic tool-use (nếu USE_FUNCTION_CALLING=True)

    Tự động dùng Vision khi câu hỏi được retrieve có đính kèm hình vẽ
    (metadata.has_image = true). Không cần thay đổi gì ở phía gọi agent.
    """

    def __init__(self, model: Optional[str] = None):
        self.model_name = model or settings.LLM_MODEL
        temp = compatible_temperature(self.model_name, 0.3)
        
        self.llm = ChatOpenAI(
            model=self.model_name,
            api_key=settings.OPENAI_API_KEY,
            temperature=temp,
        )
        vision_temp = compatible_temperature(settings.VISION_LLM_MODEL, 0.3)
        self.vision_llm = ChatOpenAI(
            model=settings.VISION_LLM_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=vision_temp,
        )
        # OpenAI async client — dùng bởi AgenticTeacherMixin
        self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

        # ── GraphRAG Retriever (replaces plain UnifiedRetriever) ─────
        self.graph_retriever = GraphRAGRetriever(k=5)

        # ── Reflection Engine (Test-Time Compute, fallback khi FC tắt) ──
        self.reflection = ReflectionEngine(max_corrections=1)

        # ── Langfuse Prompt Management ─────────────────────────────────
        # Fetch system prompts từ Langfuse Cloud khi khởi động.
        # Nếu Langfuse disabled hoặc lỗi network → tự động dùng fallback hardcode.
        # Điều này cho phép Data Scientist chỉnh sửa prompt trên Langfuse UI
        # mà không cần commit code hay deploy lại server.
        from app.utils.langfuse_client import get_prompt
        self._prompt_socratic = get_prompt(
            "aitutor-socratic-v1",
            fallback=TEACHER_SYSTEM_PROMPT_SOCRATIC,
        )
        self._prompt_exam = get_prompt(
            "aitutor-exam-v1",
            fallback=TEACHER_SYSTEM_PROMPT_EXAM,
        )
        self._prompt_answer = get_prompt(
            "aitutor-answer-v1",
            fallback=TEACHER_SYSTEM_PROMPT_ANSWER,
        )
        logger.debug(
            "TeacherAgent prompts loaded: socratic=%d chars, exam=%d chars, answer=%d chars",
            len(self._prompt_socratic), len(self._prompt_exam), len(self._prompt_answer),
        )

    def _prepare_prompt_context(
        self,
        search_query: str,
        *,
        skill_id: Optional[str],
        skill_ids: Optional[list[str]],
        formula_ids: Optional[list[str]],
        masteries: Optional[dict[str, float]],
        p_mastery: float,
        prerequisite_gaps: Optional[list[dict]],
        mode: Optional[str] = None,
    ) -> TeacherPromptContext:
        """Retrieve RAG context and derive the shared pedagogical metadata."""
        rag_result = self.graph_retriever.retrieve(
            query=search_query,
            skill_id=skill_id,
            masteries=masteries or {},
        )
        chapter = SKILLS.get(skill_id, {}).get("chapter") if skill_id else None
        gaps = rag_result.build_gap_warning()
        if not gaps and prerequisite_gaps:
            gap_names = ", ".join(gap["skill_name"] for gap in prerequisite_gaps)
            gaps = (
                "\n⚠️ HỌC SINH CÒN YẾU CÁC KIẾN THỨC NỀN: "
                f"{gap_names}. Hãy nhắc nhở ôn lại."
            )

        normalized_skill_ids = normalize_skill_ids(skill_ids, skill_id)
        return TeacherPromptContext(
            rag_result=rag_result,
            context=rag_result.build_context_text(),
            skill_names=format_skill_names(normalized_skill_ids),
            formulas=format_formulas(formula_ids),
            few_shot=build_few_shot_prompt(p_mastery, chapter, mode=mode),
            gaps=gaps,
        )

    def _build_system_prompt(
        self,
        prepared: TeacherPromptContext,
        *,
        mode: str,
        mastery_level: str,
        question_type: Optional[str] = None,
    ) -> str:
        """Render the selected prompt with one consistent substitution path."""
        template = {
            "exam": self._prompt_exam,
            "answer": self._prompt_answer,
        }.get(mode, self._prompt_socratic)
        replacements = {
            "{context}": prepared.context,
            "{skill_names_list}": prepared.skill_names,
            "{formulas_list}": prepared.formulas,
            "{mastery_level}": mastery_level,
            "{prerequisite_gaps}": prepared.gaps,
            "{few_shot_block}": prepared.few_shot,
        }
        for placeholder, value in replacements.items():
            template = template.replace(placeholder, value or "")

        format_instruction = answer_format_instruction(question_type)
        if format_instruction:
            template = f"{template}\n\n{format_instruction}"
        return template

    @staticmethod
    def _history_messages(chat_history: Optional[list[dict]]) -> list:
        """Convert the bounded chat history to LangChain messages."""
        messages = []
        for item in (chat_history or [])[-MAX_HISTORY_MESSAGES:]:
            if item.get("role") == "user":
                messages.append(HumanMessage(content=item["content"]))
            elif item.get("role") == "assistant":
                messages.append(AIMessage(content=item["content"]))
        return messages

    async def _extract_visual_evidence(
        self,
        images: list[tuple[bytes, str, str]],
        ocr_text: str,
        user_text: str,
    ) -> str:
        """Run a focused perception pass before asking the VLM to solve."""
        content: list[dict] = [{
            "type": "text",
            "text": (
                "OCR/câu hỏi chỉ dùng để định hướng vùng cần đọc; ảnh mới là nguồn "
                f"bằng chứng chính.\n\nOCR:\n{ocr_text[:3000]}\n\nYêu cầu:\n{user_text[:1000]}"
            ),
        }]
        for index, (raw, mime_type, label) in enumerate(images[:4], start=1):
            encoded = base64.b64encode(raw).decode("utf-8")
            content.extend([
                {"type": "text", "text": f"[Hình {index} — loại dự kiến: {label}]"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{encoded}",
                        "detail": "high",
                    },
                },
            ])

        try:
            response = await self.vision_llm.ainvoke([
                SystemMessage(content=VISUAL_EVIDENCE_PROMPT),
                HumanMessage(content=content),
            ])
            log_from_response(
                agent="Teacher-VisualEvidence",
                model=settings.VISION_LLM_MODEL,
                response=response,
                extra=f"images={len(images[:4])}",
            )
            return str(response.content).strip()
        except Exception as exc:
            # The main vision solve still has the original images, so this pass
            # is an accuracy enhancement rather than a new failure point.
            logger.warning("Visual evidence extraction failed: %s", exc)
            return ""

    async def respond(
        self,
        question: str,
        mode: str = "socratic",
        mastery_level: str = "beginner",
        prerequisite_gaps: Optional[list[dict]] = None,
        chat_history: Optional[list[dict]] = None,
        skill_id: Optional[str] = None,
        skill_ids: Optional[list[str]] = None,
        formula_ids: Optional[list[str]] = None,
        masteries: Optional[dict[str, float]] = None,
        p_mastery: float = 0.1,
        trace_id: Optional[str] = None,
        question_type: Optional[str] = None,
    ) -> str:
        """Generate adaptive teaching response.

        Enhanced pipeline:
          1. GraphRAG retrieval (skill-aware + prerequisite enrichment)
          2. Dynamic Few-Shot prompt injection (BKT-driven)
          3. LLM generation
          4. Reflection loop (SymPy verification + self-correction)
        """
        # ── Step 1: GraphRAG Retrieval ───────────────────────────
        from app.utils.langfuse_client import new_span, end_span
        rag_span = new_span(
            name="rag.graph_retrieval",
            trace_id=trace_id,
            input_data={"query": question[:300], "skill_id": skill_id},
        )
        # Retrieval + rerank là CPU/IO blocking — chạy trong thread riêng
        # để không chặn event loop của các request khác.
        prepared = await asyncio.to_thread(
            self._prepare_prompt_context,
            question,
            skill_id=skill_id,
            skill_ids=skill_ids,
            formula_ids=formula_ids,
            masteries=masteries,
            p_mastery=p_mastery,
            prerequisite_gaps=prerequisite_gaps,
            mode=mode,
        )
        rag_result = prepared.rag_result
        end_span(rag_span, output={
            "num_main_docs": len(rag_result.main_docs),
            "num_prereq_docs": len(rag_result.prereq_docs),
            "weak_skills": [w["skill_id"] for w in rag_result.weak_skills],
            "has_images": has_images(rag_result.all_docs),
            "search_queries": rag_result.search_queries,
        })
        system_prompt = self._build_system_prompt(
            prepared,
            mode=mode,
            mastery_level=mastery_level,
            question_type=question_type,
        )
        messages = [SystemMessage(content=system_prompt), *self._history_messages(chat_history)]

        # ── Build HumanMessage: text-only hoặc multimodal ──────────
        all_docs = rag_result.all_docs
        if has_images(all_docs):
            human_content = build_multimodal_content(
                text=question,
                retrieved_docs=all_docs,
                max_images=3,
            )
            human_message = HumanMessage(content=human_content)
        else:
            human_message = HumanMessage(content=question)

        messages.append(human_message)

        # ── Step 4: LLM Generation — Hybrid routing ──────────────────────────
        # Route heavy-math questions to Agentic Tool-Calling;
        # use cheaper Reflection pipeline for all other questions.
        require_tool = requires_math_tool(question, skill_id)
        use_agentic = (
            settings.USE_FUNCTION_CALLING  # global override (e.g. for A/B testing)
            or require_tool
        )

        if use_agentic:
            # Agentic path: LLM → tool call → … → final answer
            oai_messages = _normalize_messages(messages)
            logger.info("🧠 TeacherAgent [Agentic] skill=%s", skill_id)
            agentic_answer = await self.respond_agentic(
                messages=oai_messages,
                question=question,
                skill_id=skill_id,
                mode=mode,
                trace_id=trace_id,
                require_tool=require_tool,
            )
            return ensure_answer_tag(agentic_answer, question_type)

        # Reflection path: single LLM call → post-hoc SymPy verify (cheaper)
        logger.info("📝 TeacherAgent [Reflection] skill=%s", skill_id)
        from app.utils.langfuse_client import new_generation, end_generation
        gen = new_generation(
            name="teacher.llm_call",
            model=self.model_name,
            input_text=question[:400],
            metadata={"mode": mode, "skill_id": skill_id, "routing": "reflection"},
            trace_id=trace_id,
        )

        response = await self.llm.ainvoke(messages)
        draft = response.content

        # ── Cost log ───────────────────────────────────────────────────
        img_flag = "vision" if has_images(all_docs) else "text"
        log_from_response(
            agent="Teacher",
            model=self.model_name,
            response=response,
            extra=f"mode={mode},input={img_flag},graphrag=true,few_shot={get_mastery_tier(p_mastery)},routing=reflection",
        )

        # Kết thúc Langfuse generation span
        usage = getattr(response, "usage_metadata", None)
        end_generation(
            gen,
            output=draft[:500],
            input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
        )

        # ── Step 5: Reflection — verify math with SymPy ────────────────
        reflection_result = await self.reflection.reflect(
            draft=draft,
            question=question,
        )

        if reflection_result.was_corrected:
            logger.info("🔄 Reflection corrected the answer (skill=%s)", skill_id)

        final_response = reflection_result.build_full_response(include_thinking=True)
        return ensure_answer_tag(final_response, question_type)

    async def respond_stream(
        self,
        question: str,
        mode: str = "socratic",
        mastery_level: str = "beginner",
        prerequisite_gaps: Optional[list[dict]] = None,
        chat_history: Optional[list[dict]] = None,
        skill_id: Optional[str] = None,
        skill_ids: Optional[list[str]] = None,
        formula_ids: Optional[list[str]] = None,
        masteries: Optional[dict[str, float]] = None,
        p_mastery: float = 0.1,
    ):
        """Async generator: stream tokens từng mảnh từ LLM (dùng cho SSE endpoint).

        Yields từng chunk string (token).  Caller dùng `async for token in respond_stream(...)`.

        NOTE: Reflection/SymPy verify bị bỏ qua trong stream mode — đây là
        trade-off chấp nhận được vì UX streaming quan trọng hơn với path này.
        """
        # ── Step 1: GraphRAG (cần xong trước khi bắt đầu stream) ──
        # Chạy trong thread riêng để không chặn event loop.
        prepared = await asyncio.to_thread(
            self._prepare_prompt_context,
            question,
            skill_id=skill_id,
            skill_ids=skill_ids,
            formula_ids=formula_ids,
            masteries=masteries,
            p_mastery=p_mastery,
            prerequisite_gaps=prerequisite_gaps,
            mode=mode,
        )
        system_prompt = self._build_system_prompt(
            prepared,
            mode=mode,
            mastery_level=mastery_level,
        )

        # ── Step 4: Build OpenAI messages (dùng AsyncOpenAI vì LangChain không hỗ trợ stream=True dễ) ──
        oai_messages: list[dict] = [{"role": "system", "content": system_prompt}]
        if chat_history:
            for h in chat_history[-MAX_HISTORY_MESSAGES:]:
                if h.get("role") in ("user", "assistant"):
                    # Truncate content quá dài trong lịch sử
                    content = h["content"]
                    if isinstance(content, str) and len(content) > 2000:
                        content = content[:2000] + "..."
                    oai_messages.append({"role": h["role"], "content": content})
        oai_messages.append({"role": "user", "content": question})

        # ── Step 5: Stream tokens ───────────────────────────────────────
        temp = compatible_temperature(self.model_name, 0.3)
        extra_kwargs = {}
        # TTFT: câu không cần tính toán nặng → hạ reasoning effort để token
        # đầu ra nhanh (đo được ~22s → mục tiêu <10s). Bài heavy-math giữ
        # effort mặc định vì cần suy luận đủ sâu để gợi ý/giải đúng.
        if is_reasoning_model(self.model_name) and not requires_math_tool(question, skill_id):
            extra_kwargs["reasoning_effort"] = "low"
        stream = await self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=oai_messages,
            temperature=temp,
            stream=True,
            **extra_kwargs,
        )

        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    async def respond_with_image(
        self,
        image_bytes: bytes,
        image_mime: str,
        ocr_text: str,
        user_text: str = "",
        mode: str = "socratic",
        mastery_level: str = "beginner",
        prerequisite_gaps: Optional[list[dict]] = None,
        chat_history: Optional[list[dict]] = None,
        skill_id: Optional[str] = None,
        skill_ids: Optional[list[str]] = None,
        formula_ids: Optional[list[str]] = None,
        masteries: Optional[dict[str, float]] = None,
        p_mastery: float = 0.1,
        question_type: Optional[str] = None,
        image_label: str = "figure",
        additional_images: Optional[list[tuple[bytes, str, str]]] = None,
    ) -> str:
        """
        Hybrid Vision: gửi ảnh gốc + LaTeX OCR + câu hỏi user vào LLM trong một call.

        Enhanced with GraphRAG + Dynamic Few-Shot + Reflection.
        """
        chat_history = chat_history or []

        # ── GraphRAG: search bằng ocr_text (LaTeX) ─────────────────────
        search_query = ocr_text if ocr_text else (user_text or "bài toán")
        prepared = await asyncio.to_thread(
            self._prepare_prompt_context,
            search_query,
            skill_id=skill_id,
            skill_ids=skill_ids,
            formula_ids=formula_ids,
            masteries=masteries,
            p_mastery=p_mastery,
            prerequisite_gaps=prerequisite_gaps,
            mode=mode,
        )
        system_prompt = self._build_system_prompt(
            prepared,
            mode=mode,
            mastery_level=mastery_level,
            question_type=question_type,
        )
        messages = [SystemMessage(content=system_prompt), *self._history_messages(chat_history)]

        # ── Build HumanMessage: focused images + OCR + user text ───────
        image_inputs = [(image_bytes, image_mime, image_label)]
        image_inputs.extend((additional_images or [])[:3])

        visual_evidence = ""
        if needs_visual_feature_pass(
            ocr_text,
            user_text,
            [label for _, _, label in image_inputs],
        ):
            visual_evidence = await self._extract_visual_evidence(
                image_inputs,
                ocr_text,
                user_text,
            )

        text_parts = []
        if ocr_text and not ocr_text.startswith("["):
            text_parts.append(f"[Nội dung bài toán từ ảnh (LaTeX)]\n{ocr_text}")
        if user_text:
            text_parts.append(f"[Câu hỏi của học sinh]\n{user_text}")
        if visual_evidence:
            text_parts.append(
                "[Bằng chứng từ lượt đọc hình độc lập — cần đối chiếu lại với ảnh]\n"
                f"{visual_evidence}"
            )
        if not text_parts:
            text_parts.append("Hãy giải giúp em bài toán trong ảnh trên.")

        human_content: list[dict] = []
        for index, (raw, mime_type, label) in enumerate(image_inputs, start=1):
            b64 = base64.b64encode(raw).decode("utf-8")
            human_content.extend([{
                "type": "text",
                "text": f"[Hình {index} — {label}]",
            }, {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{b64}",
                    "detail": "high",
                },
            }])
        human_content.append({
            "type": "text",
            "text": "\n\n".join(text_parts),
        })
        messages.append(HumanMessage(content=human_content))

        tool_question = f"{ocr_text}\n{user_text}".strip() or search_query
        if requires_math_tool(tool_question, skill_id):
            logger.info("🧠 TeacherAgent [Vision+Agentic] skill=%s", skill_id)
            draft = await self.respond_agentic(
                messages=_normalize_messages(messages),
                question=tool_question,
                skill_id=skill_id,
                mode=mode,
                require_tool=True,
                model_override=settings.VISION_LLM_MODEL,
            )
            return ensure_answer_tag(draft, question_type)

        response = await self.vision_llm.ainvoke(messages)
        draft = response.content

        # ── Cost log ───────────────────────────────────────────────
        log_from_response(
            agent="Teacher",
            model=settings.VISION_LLM_MODEL,
            response=response,
            extra=f"mode={mode},input=hybrid_vision,graphrag=true,few_shot={get_mastery_tier(p_mastery)}",
        )

        # ── Reflection — verify math with SymPy ───────────────────
        reflection_result = await self.reflection.reflect(
            draft=draft,
            question=tool_question,
        )

        final_response = reflection_result.build_full_response(include_thinking=True)
        return ensure_answer_tag(final_response, question_type)
