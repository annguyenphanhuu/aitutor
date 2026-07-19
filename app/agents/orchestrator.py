"""Orchestrator — routes requests to the appropriate agent."""

from typing import Optional
from dataclasses import dataclass
from sqlalchemy.ext.asyncio import AsyncSession
from openai import AsyncOpenAI

from app.config import get_settings
from app.agents.teacher_agent import TeacherAgent
from app.agents.assessor_agent import AssessorAgent
from app.agents.planner_agent import PlannerAgent
from app.agents.visualizer_agent import VisualizerAgent
from app.knowledge_tracing.service import (
    get_all_masteries,
    get_mastery_profile,
    update_mastery,
    identify_gaps,
)
from app.knowledge_tracing.bkt import BKTModel
from app.knowledge_tracing.skill_graph import SKILLS
import json

settings = get_settings()

CLASSIFIER_PROMPT = """Phân loại ý định của học sinh thành một trong các loại sau:
- "explain": Hỏi bài, cần giải thích lý thuyết hoặc hướng dẫn giải từng bước
- "answer": Chỉ muốn biết đáp án / kết quả của bài toán (ví dụ: "đáp án là gì?", "kết quả bằng bao nhiêu?", "cho tôi xem đáp án")
- "assess": Gửi câu trả lời của mình để kiểm tra đúng/sai
- "plan": Hỏi về kế hoạch học tập, nên ôn gì
- "quiz": Muốn làm bài kiểm tra, luyện tập, ra đề (ví dụ: "cho em làm quiz", "ra đề thi đạo hàm", "cho bài tập luyện tập")
- "review": Muốn ôn bài cũ, ôn tập lại (ví dụ: "ôn tập", "nhắc lại kiến thức cũ", "có gì cần ôn không")
- "diagnostic": Muốn kiểm tra đầu vào, đánh giá năng lực (ví dụ: "test đầu vào", "đánh giá năng lực", "kiểm tra trình độ")
- "visualize": Muốn xem đồ thị, biểu đồ, hình ảnh toán học (ví dụ: "vẽ đồ thị", "đồ thị hàm số", "minh họa hình học")
- "off_topic": Câu hỏi/yêu cầu KHÔNG liên quan đến Toán học (ví dụ: tư vấn tình cảm, hỏi thời tiết, chuyện phiếm, chào hỏi đơn thuần, yêu cầu làm việc khác ngoài Toán)

Đồng thời xác định các kỹ năng Toán 12 liên quan (skill_ids) và công thức liên quan (formula_ids).
Danh sách skill_id:
{skills_list}

Danh sách formula_id:
{formulas_list}

Trả về JSON:
{{"intent": "explain|answer|assess|plan|quiz|review|diagnostic|visualize|off_topic", "skill_id": "skill_id_chinh_hoac_null", "skill_ids": ["skill_id_1"], "formula_ids": ["formula_id_1"], "is_answer_submission": true/false}}

CHỈ TRẢ VỀ JSON.
"""


@dataclass(frozen=True)
class RoutingContext:
    """Normalized classifier and mastery data shared by all entry points."""

    classification: dict
    intent: str
    skill_id: Optional[str]
    skill_ids: list[str]
    formula_ids: list[str]
    masteries: dict[str, float]
    current_mastery: float
    mastery_level: str
    mode: str


class Orchestrator:
    """Routes student messages to the correct agent."""

    def __init__(self):
        self.teacher = TeacherAgent()
        self.assessor = AssessorAgent()
        self.planner = PlannerAgent()
        self.visualizer = VisualizerAgent()
        self.bkt = BKTModel()
        # OpenAI Responses API client (hỗ trợ reasoning parameter cho gpt-5.4)
        self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async def classify_intent(self, message: str, trace_id: Optional[str] = None) -> dict:
        """Classify student message intent and related skill."""
        from app.utils.langfuse_client import new_generation, end_generation
        skills_list = "\n".join(
            [f"- {k}: {v['name']} ({v['chapter']})" for k, v in SKILLS.items()]
        )
        try:
            from app.rag.formula_registry import list_formulas

            formulas_list = "\n".join(
                f"- {formula['id']}: {formula.get('metadata', {}).get('chapter', '')}"
                for formula in list_formulas()
            )
        except Exception:
            formulas_list = ""

        prompt = CLASSIFIER_PROMPT.format(
            skills_list=skills_list,
            formulas_list=formulas_list,
        )

        # ── Langfuse generation span ───────────────────────────────────────
        gen = new_generation(
            name="orchestrator.classify_intent",
            model=settings.LLM_MODEL_MINI,
            input_text=message[:300],
            trace_id=trace_id,
        )

        # MINI tier: phân loại intent là task đơn giản, không cần reasoning sâu
        response = await self.openai_client.responses.create(
            model=settings.LLM_MODEL_MINI,
            input=[
                {
                    "role": "user",
                    "content": f"{prompt}\n\nTin nhắn học sinh: {message}",
                }
            ],
        )

        # ── cost log ──────────────────────────────────────────────────────────
        input_tokens  = getattr(response.usage, "input_tokens",  0) if response.usage else 0
        output_tokens = getattr(response.usage, "output_tokens", 0) if response.usage else 0
        from app.utils.cost_tracker import log_call
        log_call(agent="Classifier", model=settings.LLM_MODEL_MINI,
                 input_tokens=input_tokens, output_tokens=output_tokens)

        try:
            content = response.output_text.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                content = content.rsplit("```", 1)[0]
            result = json.loads(content)
        except (json.JSONDecodeError, IndexError):
            result = {
                "intent": "explain",
                "skill_id": None,
                "skill_ids": [],
                "formula_ids": [],
                "is_answer_submission": False,
            }

        skill_ids = result.get("skill_ids") or []
        if not skill_ids and result.get("skill_id"):
            skill_ids = [result["skill_id"]]
        result["skill_ids"] = skill_ids
        result["skill_id"] = result.get("skill_id") or (skill_ids[0] if skill_ids else None)
        result["formula_ids"] = result.get("formula_ids") or []

        # ── Kết thúc generation span ──────────────────────────────────────
        end_generation(
            gen,
            output=json.dumps(result),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        return result

    async def _prepare_routing_context(
        self,
        db: AsyncSession,
        message: str,
        mode: str,
        user_id: int,
        trace_id: Optional[str] = None,
    ) -> RoutingContext:
        """Classify once and normalize mastery/mode for a request."""
        classification = await self.classify_intent(message, trace_id=trace_id)
        skill_id = classification.get("skill_id")
        skill_ids = classification.get("skill_ids") or ([skill_id] if skill_id else [])
        formula_ids = classification.get("formula_ids") or []
        if not skill_id and skill_ids:
            skill_id = skill_ids[0]

        masteries = await get_all_masteries(db, user_id)
        current_mastery = masteries.get(skill_id, 0.1) if skill_id else 0.1
        resolved_mode = (
            "exam" if current_mastery >= 0.7 else "socratic"
        ) if mode == "auto" else mode
        return RoutingContext(
            classification=classification,
            intent=classification.get("intent", "explain"),
            skill_id=skill_id,
            skill_ids=skill_ids,
            formula_ids=formula_ids,
            masteries=masteries,
            current_mastery=current_mastery,
            mastery_level=self.bkt.get_mastery_level(current_mastery),
            mode=resolved_mode,
        )

    async def handle_message(
        self,
        db: AsyncSession,
        message: str,
        mode: str = "auto",
        chat_history: Optional[list[dict]] = None,
        user_id: int = 1,
        langfuse_trace=None,     # Optional Langfuse trace từ routes.py
        _routing_context: Optional[RoutingContext] = None,
        **_,                     # session_id... — chỉ engine LangGraph dùng
    ) -> dict:
        """
        Main entry point: classify intent and route to the right agent.

        Returns dict with response, skill_id, mastery, mode_used.
        """
        chat_history = chat_history or []
        # Lấy trace_id để truyền xuống các span con
        trace_id: Optional[str] = getattr(langfuse_trace, "id", None)

        routing = _routing_context or await self._prepare_routing_context(
            db, message, mode, user_id, trace_id
        )
        classification = routing.classification
        intent = routing.intent
        skill_id = routing.skill_id
        skill_ids = routing.skill_ids
        formula_ids = routing.formula_ids
        masteries = routing.masteries
        current_mastery = routing.current_mastery
        mastery_level = routing.mastery_level
        mode = routing.mode

        # Step 3: Route to agent
        if intent == "off_topic":
            response_text = (
                "Thầy hiểu em đang có chuyện bên ngoài, và điều đó hoàn toàn bình thường. 😊\n\n"
                "Nhưng thầy là gia sư Toán 12 — thầy chỉ có thể đồng hành cùng em trên con đường chinh phục Toán thôi nhé.\n\n"
                "Nếu em đang căng thẳng vì học hành, thầy rất sẵn sàng giúp em:\n"
                "- 📋 Lập **kế hoạch ôn tập** phù hợp để bớt áp lực\n"
                "- 🔍 **Chẩn đoán năng lực** để biết mình đang ở đâu\n"
                "- 💡 Giải thích những phần Toán em chưa hiểu\n\n"
                "Em muốn bắt đầu từ đâu?"
            )
            return {
                "response": response_text,
                "skill_id": None,
                "skill_name": None,
                "mastery_level": None,
                "mode_used": "off_topic",
            }

        elif intent == "plan":
            profile = await get_mastery_profile(db, user_id)
            plan = await self.planner.create_plan(profile)
            response_text = self._format_plan(plan)
            return {
                "response": response_text,
                "skill_id": None,
                "skill_name": None,
                "mastery_level": None,
                "mode_used": "plan",
            }

        elif intent == "quiz":
            # Student wants to practice with quiz
            target_skill = skill_id or "derivative_basic"
            skill_info = SKILLS.get(target_skill, {})
            difficulty = 2 if current_mastery >= 0.5 else 1
            response_text = (
                f"📝 **Bài kiểm tra: {skill_info.get('name', target_skill)}** (Độ khó: {difficulty})\n\n"
                f"Sử dụng nút **📝 Làm quiz** trên giao diện để bắt đầu, "
                f"hoặc gọi API: `POST /api/quiz/generate` với skill_id=`{target_skill}`.\n\n"
                f"💡 Mình sẽ tự động đánh giá và cập nhật năng lực của em sau mỗi câu hỏi!"
            )
            return {
                "response": response_text,
                "skill_id": target_skill,
                "skill_name": skill_info.get("name"),
                "mastery_level": round(current_mastery, 3),
                "mode_used": "quiz",
            }

        elif intent == "review":
            response_text = (
                "📖 **Ôn tập chống quên lãng**\n\n"
                "Hệ thống sẽ kiểm tra các kiến thức em có nguy cơ quên.\n"
                "Sử dụng nút **📖 Ôn tập** hoặc gọi API: `GET /api/review/due` để lấy thẻ ôn tập.\n\n"
                "💡 Thuật toán SM-2 (SuperMemo) sẽ tự lên lịch ôn tập cá nhân hóa!"
            )
            return {
                "response": response_text,
                "skill_id": None,
                "skill_name": None,
                "mastery_level": None,
                "mode_used": "review",
            }

        elif intent == "diagnostic":
            response_text = (
                "🔍 **Test Chẩn Đoán Năng Lực**\n\n"
                "Bài test này gồm 12 câu hỏi từ tất cả các chương, "
                "giúp đánh giá điểm mạnh/yếu của em.\n\n"
                "Sử dụng nút **🔍 Chẩn đoán** hoặc gọi API: `POST /api/diagnostic/start`.\n\n"
                "⏱️ Thời gian dự kiến: khoảng 10-15 phút."
            )
            return {
                "response": response_text,
                "skill_id": None,
                "skill_name": None,
                "mastery_level": None,
                "mode_used": "diagnostic",
            }

        elif intent == "visualize":
            # Extract math expression from the message, with chat history context
            vis_data = await self._extract_and_visualize(message, chat_history=chat_history)
            skill_info = SKILLS.get(skill_id, {})
            response_text = (
                "📊 Đồ thị đã được tạo! Xem bên dưới."
            )
            if vis_data is None:
                response_text = (
                    "⚠️ Không tìm được biểu thức toán học trong tin nhắn. "
                    "Em hãy ghi rõ hàm số cần vẽ, ví dụ: *vẽ đồ thị y = 2x\u00b7sin(x) + x\u00b2\u00b7cos(x)*."
                )
            elif vis_data.get("vis_type") == "error":
                response_text = f"⚠️ {vis_data['data'].get('message', 'Không thể vẽ đồ thị.')}"
                vis_data = None

            return {
                "response": response_text,
                "skill_id": skill_id,
                "skill_name": skill_info.get("name"),
                "mastery_level": round(current_mastery, 3),
                "mode_used": "visualize",
                "visualization": vis_data,
            }

        elif intent == "answer":
            # Student just wants the answer — teach in direct answer mode
            prereq_gaps = []
            if skill_id:
                prereq_gaps = await identify_gaps(db, skill_id, user_id)

            response_text = await self.teacher.respond(
                question=message,
                mode="answer",
                mastery_level=mastery_level,
                prerequisite_gaps=prereq_gaps,
                chat_history=chat_history,
                skill_id=skill_id,
                skill_ids=skill_ids,
                formula_ids=formula_ids,
                masteries=masteries,
                p_mastery=current_mastery,
                trace_id=trace_id,
            )

            skill_info = SKILLS.get(skill_id, {})
            return {
                "response": response_text,
                "skill_id": skill_id,
                "skill_ids": skill_ids,
                "formula_ids": formula_ids,
                "skill_name": skill_info.get("name"),
                "mastery_level": round(current_mastery, 3),
                "mode_used": "answer",
            }

        elif intent == "assess" and classification.get("is_answer_submission"):
            original_question = self._find_original_question(chat_history, message)

            assessment = await self.assessor.assess(
                question=original_question,
                student_answer=message,
                skill_ids=skill_ids,
                formula_ids=formula_ids,
                chat_history=chat_history,
            )

            # Update mastery based on correctness
            new_mastery = current_mastery
            assessed = assessment.get("skills_assessed") or {
                sid: {"passed": assessment["is_correct"]}
                for sid in skill_ids
            }
            if assessed:
                mastery_updates = await update_mastery(db, assessed, user_id=user_id)
                if isinstance(mastery_updates, dict):
                    new_mastery = mastery_updates.get(skill_id, current_mastery)
                else:
                    new_mastery = mastery_updates

            skill_info = SKILLS.get(skill_id, {})
            return {
                "response": self._format_assessment(assessment),
                "skill_id": skill_id,
                "skill_ids": skill_ids,
                "formula_ids": formula_ids,
                "skill_name": skill_info.get("name"),
                "mastery_level": round(new_mastery, 3),
                "mode_used": "assess",
            }

        else:
            # Default: teach/explain
            prereq_gaps = []
            if skill_id:
                prereq_gaps = await identify_gaps(db, skill_id, user_id)

            # Check if the message contains a graph request embedded in explanation
            vis_data = None
            graph_keywords = ["đồ thị", "vẽ", "biểu đồ", "minh họa", "hình ảnh"]
            if any(kw in message.lower() for kw in graph_keywords):
                vis_data = await self._extract_and_visualize(message, chat_history=chat_history)
                if vis_data and vis_data.get("vis_type") == "error":
                    vis_data = None

            response_text = await self.teacher.respond(
                question=message,
                mode=mode,
                mastery_level=mastery_level,
                prerequisite_gaps=prereq_gaps,
                chat_history=chat_history,
                skill_id=skill_id,
                skill_ids=skill_ids,
                formula_ids=formula_ids,
                masteries=masteries,
                p_mastery=current_mastery,
                trace_id=trace_id,
            )

            skill_info = SKILLS.get(skill_id, {})
            return {
                "response": response_text,
                "skill_id": skill_id,
                "skill_ids": skill_ids,
                "formula_ids": formula_ids,
                "skill_name": skill_info.get("name"),
                "mastery_level": round(current_mastery, 3),
                "mode_used": mode,
                "visualization": vis_data,
            }

    async def handle_image_message(
        self,
        db: AsyncSession,
        image_bytes: bytes,
        image_mime: str,
        ocr_text: str,
        user_text: str = "",
        mode: str = "auto",
        chat_history: Optional[list[dict]] = None,
        user_id: int = 1,
    ) -> dict:
        """
        Hybrid Vision entry point: xử lý ảnh + (optional) text user.

        Flow:
        1. Classify intent từ ocr_text + user_text
        2. Get mastery
        3. Gọi teacher.respond_with_image() — LLM nhìn ảnh gốc + LaTeX OCR + câu hỏi
        """
        chat_history = chat_history or []

        # Step 1: Classify intent từ combined text
        combined_for_classify = "\n".join(filter(None, [ocr_text, user_text])).strip()
        if not combined_for_classify:
            combined_for_classify = "bài toán trong ảnh"

        routing = await self._prepare_routing_context(
            db, combined_for_classify, mode, user_id
        )
        skill_id = routing.skill_id
        skill_ids = routing.skill_ids
        formula_ids = routing.formula_ids
        masteries = routing.masteries
        current_mastery = routing.current_mastery
        mastery_level = routing.mastery_level
        mode = routing.mode

        # Step 3: Prerequisite gaps
        prereq_gaps = []
        if skill_id:
            prereq_gaps = await identify_gaps(db, skill_id, user_id)

        # Step 4: Gọi respond_with_image (Hybrid Vision + GraphRAG + Reflection)
        response_text = await self.teacher.respond_with_image(
            image_bytes=image_bytes,
            image_mime=image_mime,
            ocr_text=ocr_text,
            user_text=user_text,
            mode=mode,
            mastery_level=mastery_level,
            prerequisite_gaps=prereq_gaps,
            chat_history=chat_history,
            skill_id=skill_id,
            skill_ids=skill_ids,
            formula_ids=formula_ids,
            masteries=masteries,
            p_mastery=current_mastery,
        )

        skill_info = SKILLS.get(skill_id, {})
        return {
            "response": response_text,
            "skill_id": skill_id,
            "skill_ids": skill_ids,
            "formula_ids": formula_ids,
            "skill_name": skill_info.get("name"),
            "mastery_level": round(current_mastery, 3),
            "mode_used": f"hybrid_vision_{mode}",
        }

    async def handle_message_stream(
        self,
        db: AsyncSession,
        message: str,
        mode: str = "auto",
        chat_history: Optional[list[dict]] = None,
        user_id: int = 1,
        **_,                     # session_id... — chỉ engine LangGraph dùng
    ):
        """Streaming variant của handle_message dùng cho SSE endpoint.

        Async generator — yields JSON strings theo thứ tự:
          1. {"type": "meta", "skill_id": ..., "mode_used": ..., ...}
          2. {"type": "token", "content": "..."}   (nhiều lần)
          3. {"type": "done", "full_response": "..."}
          Hoặc:
          {"type": "error", "message": "..."}   nếu có exception
        """
        import json as _json
        chat_history = chat_history or []

        # ── Step 1: Classify intent (blocking, bắt buộc trước khi stream) ──
        routing = await self._prepare_routing_context(db, message, mode, user_id)
        intent = routing.intent
        skill_id = routing.skill_id
        skill_ids = routing.skill_ids
        formula_ids = routing.formula_ids
        masteries = routing.masteries
        current_mastery = routing.current_mastery
        mastery_level = routing.mastery_level
        mode = routing.mode

        skill_info = SKILLS.get(skill_id, {})

        # ── Step 3: Yield meta frame TRƯỚC tokens ─────────────────────────
        yield _json.dumps({
            "type": "meta",
            "skill_id": skill_id,
            "skill_ids": skill_ids,
            "formula_ids": formula_ids,
            "skill_name": skill_info.get("name"),
            "mastery_level": round(current_mastery, 3),
            "mode_used": mode,
            "intent": intent,
        }, ensure_ascii=False)

        # ── Step 4: Non-streamable intents → gọi blocking rồi emit 1 lần ──
        if intent in ("off_topic", "plan", "quiz", "review", "diagnostic", "visualize", "assess"):
            try:
                result = await self.handle_message(
                    db=db, message=message, mode=mode,
                    chat_history=chat_history, user_id=user_id,
                    _routing_context=routing,
                )
                full = result.get("response", "")
                yield _json.dumps({"type": "token", "content": full}, ensure_ascii=False)
                # The done frame is the contract boundary for non-text response data.
                # Keep the complete result metadata here so SSE clients receive the
                # same visualization payload as clients of POST /api/chat.
                yield _json.dumps({
                    "type": "done",
                    "full_response": full,
                    "skill_id": result.get("skill_id"),
                    "skill_ids": result.get("skill_ids") or [],
                    "formula_ids": result.get("formula_ids") or [],
                    "skill_name": result.get("skill_name"),
                    "mastery_level": result.get("mastery_level"),
                    "mode_used": result.get("mode_used", mode),
                    "visualization": result.get("visualization"),
                }, ensure_ascii=False)
            except Exception as e:
                yield _json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False)
            return

        # ── Step 5: Stream teacher response ───────────────────────────────
        prereq_gaps = []
        if skill_id:
            prereq_gaps = await identify_gaps(db, skill_id, user_id)

        # Chọn mode cho intent "answer"
        if intent == "answer":
            mode = "answer"

        full_response = ""
        try:
            async for token in self.teacher.respond_stream(
                question=message,
                mode=mode,
                mastery_level=mastery_level,
                prerequisite_gaps=prereq_gaps,
                chat_history=chat_history,
                skill_id=skill_id,
                skill_ids=skill_ids,
                formula_ids=formula_ids,
                masteries=masteries,
                p_mastery=current_mastery,
            ):
                full_response += token
                yield _json.dumps({"type": "token", "content": token}, ensure_ascii=False)

        except Exception as e:
            logger = __import__("logging").getLogger(__name__)
            logger.error("handle_message_stream error: %s", e)
            yield _json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False)
            return

        yield _json.dumps({"type": "done", "full_response": full_response}, ensure_ascii=False)

    @staticmethod
    def _is_assessment_output(content: str) -> bool:
        """Detect a previous grading message produced by _format_assessment."""
        head = content.lstrip()[:40]
        return head.startswith(("✅", "❌")) and "Điểm:" in head

    @classmethod
    def _find_original_question(cls, chat_history: list[dict], fallback: str) -> str:
        """Recover the original problem statement for grading.

        Previous grading outputs must be skipped: after one assessment turn,
        the most recent assistant message is a score card, not the problem —
        grading against it produced wrong verdicts in multi-turn Socratic flows.
        """
        for h in reversed(chat_history):
            if h.get("role") != "assistant":
                continue
            content = h.get("content") or ""
            if cls._is_assessment_output(content):
                continue
            return content
        return fallback

    def _format_plan(self, plan: dict) -> str:
        """Format study plan into readable text."""
        lines = [f"📋 **{plan.get('summary', 'Kế hoạch học tập')}**\n"]

        priorities = plan.get("priorities", [])
        for i, p in enumerate(priorities, 1):
            urgency_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(p.get("urgency", "medium"), "⚪")
            lines.append(
                f"{i}. {urgency_emoji} **{p.get('skill_name', '')}**: "
                f"{p.get('action', '')} ({p.get('exercises', 5)} bài)"
            )

        encouragement = plan.get("encouragement", "")
        if encouragement:
            lines.append(f"\n💪 {encouragement}")

        return "\n".join(lines)

    def _format_assessment(self, assessment: dict) -> str:
        """Format assessment result into readable text."""
        emoji = "✅" if assessment.get("is_correct") else "❌"
        score = assessment.get("score", 0)

        lines = [
            f"{emoji} **Điểm: {score:.0%}**\n",
            f"**Nhận xét:** {assessment.get('feedback', '')}\n",
        ]

        if not assessment.get("is_correct"):
            error_labels = {
                "calculation": "Lỗi tính toán",
                "conceptual": "Hiểu sai khái niệm",
                "procedural": "Sai phương pháp giải",
            }
            error_type = assessment.get("error_type", "unknown")
            lines.append(f"**Loại lỗi:** {error_labels.get(error_type, error_type)}\n")

        lines.append(f"**Lời giải đúng:**\n{assessment.get('correct_solution', '')}")

        return "\n".join(lines)

    # ── Visualization helpers ──────────────────────────────────────────────

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
        self,
        message: str,
        chat_history: list[dict] | None = None,
    ) -> dict | None:
        """Use LLM to extract a plottable expression, then visualize it.

        Returns a vis_data dict or None if no expression found / not plottable.
        """
        expr = await self._extract_plottable_expr_llm(message, chat_history or [])
        if not expr:
            return None
        result = self.visualizer.generate_function_plot(expr)
        return result

    async def _extract_plottable_expr_llm(
        self,
        message: str,
        chat_history: list[dict],
    ) -> str | None:
        """Ask the LLM (cheapest model) to extract the plottable math expression."""
        from app.utils.cost_tracker import log_call

        # Build a concise history snippet (last 4 turns max)
        history_snippet = ""
        if chat_history:
            turns = chat_history[-4:]
            lines = []
            for h in turns:
                role = "Học sinh" if h["role"] == "user" else "Gia sư"
                # Truncate very long messages
                text = h["content"][:300]
                lines.append(f"{role}: {text}")
            history_snippet = "\n".join(lines)

        user_content = f"Tin nhắn học sinh: {message}"
        if history_snippet:
            user_content += f"\n\nLịch sử hội thoại gần đây:\n{history_snippet}"

        try:
            response = await self.openai_client.responses.create(
                model=settings.LLM_MODEL_NANO,  # NANO: simple extraction task
                input=[
                    {"role": "system", "content": self.EXPR_EXTRACTOR_PROMPT},
                    {"role": "user",   "content": user_content},
                ],
            )
            expr = response.output_text.strip()

            # Cost tracking
            input_tokens  = getattr(response.usage, "input_tokens",  0) if response.usage else 0
            output_tokens = getattr(response.usage, "output_tokens", 0) if response.usage else 0
            log_call(agent="ExprExtractor", model=settings.LLM_MODEL_NANO,
                     input_tokens=input_tokens, output_tokens=output_tokens)

            if not expr or expr.upper() == "NONE":
                return None
            # Basic sanity check: must contain 'x'
            if "x" not in expr.lower():
                return None
            return expr

        except Exception:
            return None

