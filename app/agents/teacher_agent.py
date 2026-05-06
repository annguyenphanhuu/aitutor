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
import base64
import logging

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from openai import AsyncOpenAI
from app.config import get_settings
from app.rag.graph_rag import GraphRAGRetriever
from app.agents.reflection import ReflectionEngine, extract_thinking_block
from app.agents.few_shot_store import build_few_shot_prompt, get_mastery_tier
from app.agents.agentic_teacher import AgenticTeacherMixin, _normalize_messages
from app.utils.cost_tracker import log_from_response
from app.utils.image_loader import build_multimodal_content, has_images
from app.knowledge_tracing.skill_graph import SKILLS
from app.rag.formula_registry import get_formulas_by_ids

logger = logging.getLogger(__name__)
settings = get_settings()

# Maximum number of history messages to include in context
MAX_HISTORY_MESSAGES = 10


def _normalize_skill_ids(
    skill_id: Optional[str] = None,
    skill_ids: Optional[list[str]] = None,
) -> list[str]:
    normalized: list[str] = []
    for sid in skill_ids or []:
        if sid and sid not in normalized:
            normalized.append(sid)
    if skill_id and skill_id not in normalized:
        normalized.insert(0, skill_id)
    return normalized


def _format_skill_names(skill_ids: list[str]) -> str:
    if not skill_ids:
        return "chua xac dinh"
    return ", ".join(
        f"{SKILLS.get(skill_id, {}).get('name', skill_id)} ({skill_id})"
        for skill_id in skill_ids
    )


def _format_formulas(formula_ids: Optional[list[str]]) -> str:
    formulas = get_formulas_by_ids(formula_ids or [])
    if not formulas:
        return "Khong co cong thuc trong tam duoc gan metadata."
    return "\n\n".join(formula.get("content", "") for formula in formulas)


TEACHER_SYSTEM_PROMPT_SOCRATIC = """Bạn là một gia sư Toán 12 giỏi, theo phương pháp Socratic.
Vai trò: Dẫn dắt học sinh tự tìm ra đáp án thay vì giải thẳng.

QUY TẮC:
0. PHẠM VI: Bạn CHỈ hỗ trợ các chủ đề Toán 12 và học tập. Nếu học sinh hỏi về tình cảm, thời tiết, tin tức, hoặc bất kỳ chủ đề nào không liên quan đến Toán, hãy từ chối lịch sự trong 1-2 câu và gợi ý quay lại câu hỏi Toán.
1. KHÔNG BAO GIỜ đưa ra lời giải hoàn chỉnh ngay lập tức.
2. Đặt câu hỏi gợi mở để học sinh tự suy nghĩ từng bước.
3. Chỉ đưa ra gợi ý (hint) khi học sinh bị mắc.
4. Sử dụng kiến thức SGK Việt Nam (Kết nối tri thức / Chân trời sáng tạo / Cánh diều).
5. Nếu học sinh chưa nắm kiến thức nền tảng, yêu cầu ôn lại trước.
6. Trả lời bằng tiếng Việt, sử dụng ký hiệu toán học chuẩn.
7. Dùng LaTeX cho công thức: $...$ cho inline, $$...$$ cho block.
8. Nếu học sinh hỏi tiếp theo (ví dụ: "vậy...", "còn...", "sao lại..."), hãy hiểu dựa trên ngữ cảnh hội thoại trước đó.

QUY TRÌNH SUY LUẬN (BẮT BUỘC) — Chain of Thought:
Trước khi đưa ra bất kỳ gợi ý hay câu hỏi nào cho học sinh, bạn PHẢI suy luận từng bước trong đầu:
  Bước 1: Xác định dạng bài (trắc nghiệm, tự luận, đúng/sai, đồ thị, hình ảnh...).
  Bước 2: Liệt kê các kiến thức/công thức liên quan cần sử dụng.
  Bước 3: Nếu bài có HÌNH VẼ hoặc ĐỒ THỊ — đọc CẨN THẬN tọa độ từ hình:
           a) Đọc CHÍNH XÁC giá trị $y$ tại $x=0$ (nhìn trục tung, kiểm tra đường đồ thị
              cắt trục $y$ tại điểm nào: $y=0$? $y=1$? $y=-1$?).
           b) Đọc giá trị $y$ tại các điểm đặc biệt khác: $x=\frac{\pi}{2}$, $x=\pi$...
           c) Ghi lại ít nhất 3 cặp $(x, y)$ CỤ THỂ đọc từ hình.
           d) Đối chiếu TỪNG phương án với các điểm đã đọc để loại trừ.
           KHÔNG BAO GIỜ kết luận chỉ dựa vào "cảm giác" hình dáng chung chung.
  Bước 4: Tự giải hoàn chỉnh để biết đáp án đúng.
  Bước 5: Dựa trên đáp án đúng, thiết kế câu hỏi gợi mở dẫn dắt học sinh.

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

TEACHER_SYSTEM_PROMPT_EXAM = """Bạn là một gia sư Toán 12, chế độ luyện thi.
Vai trò: Giúp học sinh giải nhanh, làm đề thi hiệu quả.

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


TEACHER_SYSTEM_PROMPT_ANSWER = """Bạn là một gia sư Toán 12, chế độ cung cấp đáp án.
Học sinh đã yêu cầu xem đáp án trực tiếp.

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

    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.3,
        )
        # OpenAI async client — dùng bởi AgenticTeacherMixin
        self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

        # ── GraphRAG Retriever (replaces plain UnifiedRetriever) ─────
        self.graph_retriever = GraphRAGRetriever(k=5)

        # ── Reflection Engine (Test-Time Compute, fallback khi FC tắt) ──
        self.reflection = ReflectionEngine(max_corrections=1)

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
    ) -> str:
        """Generate adaptive teaching response.

        Enhanced pipeline:
          1. GraphRAG retrieval (skill-aware + prerequisite enrichment)
          2. Dynamic Few-Shot prompt injection (BKT-driven)
          3. LLM generation
          4. Reflection loop (SymPy verification + self-correction)
        """
        # ── Step 1: GraphRAG Retrieval ─────────────────────────────
        rag_result = self.graph_retriever.retrieve(
            query=question,
            skill_id=skill_id,
            masteries=masteries or {},
        )
        context = rag_result.build_context_text()
        normalized_skill_ids = _normalize_skill_ids(skill_id, skill_ids)
        skill_names_list = _format_skill_names(normalized_skill_ids)
        formulas_list = _format_formulas(formula_ids)
        graph_gap_warning = rag_result.build_gap_warning()

        # ── Step 2: Dynamic Few-Shot ──────────────────────────────
        skill_info = SKILLS.get(skill_id, {}) if skill_id else {}
        chapter = skill_info.get("chapter")
        few_shot_block = build_few_shot_prompt(p_mastery, chapter)

        # ── Merge prerequisite warnings ───────────────────────────
        gaps_text = graph_gap_warning
        if not gaps_text and prerequisite_gaps:
            # Fallback to the old-style gap list
            gaps_list = ", ".join([g["skill_name"] for g in prerequisite_gaps])
            gaps_text = f"\n⚠️ HỌC SINH CÒN YẾU CÁC KIẾN THỨC NỀN: {gaps_list}. Hãy nhắc nhở ôn lại."

        # ── Step 3: Build messages ───────────────────────────────
        if mode == "exam":
            system_prompt = TEACHER_SYSTEM_PROMPT_EXAM.format(
                context=context,
                skill_names_list=skill_names_list,
                formulas_list=formulas_list,
                mastery_level=mastery_level,
                few_shot_block=few_shot_block,
            )
        elif mode == "answer":
            system_prompt = TEACHER_SYSTEM_PROMPT_ANSWER.format(
                context=context,
                skill_names_list=skill_names_list,
                formulas_list=formulas_list,
                mastery_level=mastery_level,
                few_shot_block=few_shot_block,
            )
        else:
            system_prompt = TEACHER_SYSTEM_PROMPT_SOCRATIC.format(
                context=context,
                skill_names_list=skill_names_list,
                formulas_list=formulas_list,
                mastery_level=mastery_level,
                prerequisite_gaps=gaps_text,
                few_shot_block=few_shot_block,
            )

        messages = [SystemMessage(content=system_prompt)]

        # Inject chat history (sliding window: last N messages)
        if chat_history:
            recent_history = chat_history[-MAX_HISTORY_MESSAGES:]
            for h in recent_history:
                if h.get("role") == "user":
                    messages.append(HumanMessage(content=h["content"]))
                elif h.get("role") == "assistant":
                    messages.append(AIMessage(content=h["content"]))

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

        # ── Step 4: LLM Generation ─────────────────────────────────────────
        # Branch A: Function Calling (ReAct agentic)
        if settings.USE_FUNCTION_CALLING:
            # normalize LangChain messages → dict format cho OpenAI SDK
            oai_messages = _normalize_messages(messages)
            return await self.respond_agentic(
                messages=oai_messages,
                question=question,
                skill_id=skill_id,
                mode=mode,
            )

        # Branch B: Legacy LangChain + Reflection
        from app.utils.langfuse_client import new_generation, end_generation
        gen = new_generation(
            name="teacher.llm_call",
            model=settings.LLM_MODEL,
            input_text=question[:400],
            metadata={"mode": mode, "skill_id": skill_id},
        )

        response = await self.llm.ainvoke(messages)
        draft = response.content

        # ── Cost log ───────────────────────────────────────────────────
        img_flag = "vision" if has_images(all_docs) else "text"
        log_from_response(
            agent="Teacher",
            model=settings.LLM_MODEL,
            response=response,
            extra=f"mode={mode},input={img_flag},graphrag=true,few_shot={get_mastery_tier(p_mastery)}",
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
            logger.info(
                "🔄 Reflection corrected the answer (skill=%s)",
                skill_id,
            )

        # Return the final answer (with thinking block stripped)
        return reflection_result.build_full_response(include_thinking=True)

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
        # ── Step 1: GraphRAG (blocking — cần xong trước khi bắt đầu stream) ──
        rag_result = self.graph_retriever.retrieve(
            query=question,
            skill_id=skill_id,
            masteries=masteries or {},
        )
        context = rag_result.build_context_text()
        normalized_skill_ids = _normalize_skill_ids(skill_id, skill_ids)
        skill_names_list = _format_skill_names(normalized_skill_ids)
        formulas_list = _format_formulas(formula_ids)
        graph_gap_warning = rag_result.build_gap_warning()

        # ── Step 2: Dynamic Few-Shot ─────────────────────────────────
        skill_info = SKILLS.get(skill_id, {}) if skill_id else {}
        chapter = skill_info.get("chapter")
        few_shot_block = build_few_shot_prompt(p_mastery, chapter)

        gaps_text = graph_gap_warning
        if not gaps_text and prerequisite_gaps:
            gaps_list = ", ".join([g["skill_name"] for g in prerequisite_gaps])
            gaps_text = f"\n⚠️ HỌC SINH CÒN YẾU CÁC KIẾN THỨC NỀN: {gaps_list}. Hãy nhắc nhở ôn lại."

        # ── Step 3: Build system prompt ───────────────────────────────
        if mode == "exam":
            system_prompt = TEACHER_SYSTEM_PROMPT_EXAM.format(
                context=context, mastery_level=mastery_level, few_shot_block=few_shot_block,
                skill_names_list=skill_names_list, formulas_list=formulas_list,
            )
        elif mode == "answer":
            system_prompt = TEACHER_SYSTEM_PROMPT_ANSWER.format(
                context=context, mastery_level=mastery_level, few_shot_block=few_shot_block,
                skill_names_list=skill_names_list, formulas_list=formulas_list,
            )
        else:
            system_prompt = TEACHER_SYSTEM_PROMPT_SOCRATIC.format(
                context=context, mastery_level=mastery_level,
                skill_names_list=skill_names_list, formulas_list=formulas_list,
                prerequisite_gaps=gaps_text or "", few_shot_block=few_shot_block,
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
        stream = await self.openai_client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=oai_messages,
            temperature=0.3,
            stream=True,
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
    ) -> str:
        """
        Hybrid Vision: gửi ảnh gốc + LaTeX OCR + câu hỏi user vào LLM trong một call.

        Enhanced with GraphRAG + Dynamic Few-Shot + Reflection.
        """
        chat_history = chat_history or []

        # ── GraphRAG: search bằng ocr_text (LaTeX) ─────────────────────
        search_query = ocr_text if ocr_text else (user_text or "bài toán")
        rag_result = self.graph_retriever.retrieve(
            query=search_query,
            skill_id=skill_id,
            masteries=masteries or {},
        )
        context = rag_result.build_context_text()
        normalized_skill_ids = _normalize_skill_ids(skill_id, skill_ids)
        skill_names_list = _format_skill_names(normalized_skill_ids)
        formulas_list = _format_formulas(formula_ids)

        # ── Dynamic Few-Shot ───────────────────────────────────────────
        skill_info = SKILLS.get(skill_id, {}) if skill_id else {}
        chapter = skill_info.get("chapter")
        few_shot_block = build_few_shot_prompt(p_mastery, chapter)

        # ── Prerequisite gaps ──────────────────────────────────────
        gaps_text = rag_result.build_gap_warning()
        if not gaps_text and prerequisite_gaps:
            gaps_list = ", ".join([g["skill_name"] for g in prerequisite_gaps])
            gaps_text = f"\n⚠️ HỌC SINH CÒN YẾU CÁC KIẾN THỨC NỀN: {gaps_list}. Hãy nhắc nhở ôn lại."

        # ── Chọn system prompt theo mode ───────────────────────────
        if mode == "exam":
            system_prompt = TEACHER_SYSTEM_PROMPT_EXAM.format(
                context=context,
                skill_names_list=skill_names_list,
                formulas_list=formulas_list,
                mastery_level=mastery_level,
                few_shot_block=few_shot_block,
            )
        elif mode == "answer":
            system_prompt = TEACHER_SYSTEM_PROMPT_ANSWER.format(
                context=context,
                skill_names_list=skill_names_list,
                formulas_list=formulas_list,
                mastery_level=mastery_level,
                few_shot_block=few_shot_block,
            )
        else:
            system_prompt = TEACHER_SYSTEM_PROMPT_SOCRATIC.format(
                context=context,
                skill_names_list=skill_names_list,
                formulas_list=formulas_list,
                mastery_level=mastery_level,
                prerequisite_gaps=gaps_text,
                few_shot_block=few_shot_block,
            )

        # ── Build messages với history ─────────────────────────────
        messages = [SystemMessage(content=system_prompt)]
        if chat_history:
            recent = chat_history[-MAX_HISTORY_MESSAGES:]
            for h in recent:
                if h.get("role") == "user":
                    messages.append(HumanMessage(content=h["content"]))
                elif h.get("role") == "assistant":
                    messages.append(AIMessage(content=h["content"]))

        # ── Build HumanMessage: image + ocr_text + user_text ───────
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        image_data_url = f"data:{image_mime};base64,{b64}"

        text_parts = []
        if ocr_text and not ocr_text.startswith("["):
            text_parts.append(f"[Nội dung bài toán từ ảnh (LaTeX)]\n{ocr_text}")
        if user_text:
            text_parts.append(f"[Câu hỏi của học sinh]\n{user_text}")
        if not text_parts:
            text_parts.append("Hãy giải giúp em bài toán trong ảnh trên.")

        human_content = [
            {
                "type": "image_url",
                "image_url": {"url": image_data_url, "detail": "high"},
            },
            {
                "type": "text",
                "text": "\n\n".join(text_parts),
            },
        ]
        messages.append(HumanMessage(content=human_content))

        response = await self.llm.ainvoke(messages)
        draft = response.content

        # ── Cost log ───────────────────────────────────────────────
        log_from_response(
            agent="Teacher",
            model=settings.LLM_MODEL,
            response=response,
            extra=f"mode={mode},input=hybrid_vision,graphrag=true,few_shot={get_mastery_tier(p_mastery)}",
        )

        # ── Reflection — verify math with SymPy ───────────────────
        reflection_result = await self.reflection.reflect(
            draft=draft,
            question=search_query,
        )

        return reflection_result.build_full_response(include_thinking=True)
