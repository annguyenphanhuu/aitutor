"""Agentic Teacher Mixin — ReAct pattern với OpenAI Function Calling.

Thay thế heuristic-based Reflection Engine bằng LLM-driven tool selection.

Flow (ReAct pattern):
    Round 1:
        LLM nhận question + context → quyết định gọi tool hay trả lời ngay
        Nếu gọi compute_derivative → SymPy tính → kết quả append vào messages
    Round 2..N:
        LLM nhận tool results → gọi thêm tool hoặc generate final answer
    Final:
        LLM không emit tool_calls nữa → return msg.content

Chain-of-thought được embedded tự nhiên trong tool calls, không cần
<thinking> block hack như Reflection engine cũ.

Feature flag: USE_FUNCTION_CALLING=true trong .env để bật.
Khi false → TeacherAgent dùng Reflection engine cũ (backward compatible).
"""

from __future__ import annotations

import json
import logging
from typing import Optional, Any

from openai import AsyncOpenAI
from app.config import get_settings
from app.agents.tool_schemas import OPENAI_MATH_TOOLS, TOOL_NAMES, _get_math_tools
from app.utils.cost_tracker import log_call

logger = logging.getLogger(__name__)
settings = get_settings()


class AgenticTeacherMixin:
    """
    Mixin cung cấp respond_agentic() cho TeacherAgent.

    Yêu cầu: class chứa phải khởi tạo self.openai_client (AsyncOpenAI).
    """

    async def respond_agentic(
        self,
        messages: list[dict],
        question: str,
        skill_id: Optional[str] = None,
        mode: str = "socratic",
        trace_id: Optional[str] = None,
    ) -> str:
        """
        ReAct loop: LLM generate → tool call (nếu cần) → append result → lặp.

        Parameters
        ----------
        messages : list[dict]
            Danh sách messages theo OpenAI format (system + history + human).
        question : str
            Câu hỏi gốc của học sinh (dùng để log).
        skill_id : str | None
            skill_id để log cost.
        mode : str
            "socratic" | "exam" | "answer"
        trace_id : str | None
            Langfuse trace ID để gắn các span con vào trace cha.

        Returns
        -------
        str
            Final answer sau khi LLM không còn emit tool calls.
        """
        from app.utils.langfuse_client import new_span, end_span, new_generation, end_generation
        max_rounds = settings.FUNCTION_CALLING_MAX_ROUNDS
        math_tools = _get_math_tools()

        # Convert LangChain message objects → OpenAI dict format nếu cần
        current_messages = _normalize_messages(messages)

        # Accumulate tool call trace để log / debug
        tool_call_log: list[dict] = []

        for round_num in range(max_rounds + 1):
            # ── Langfuse generation span cho mỗi LLM round ────────────────
            agentic_gen = new_generation(
                name=f"teacher.agentic_round_{round_num}",
                model=settings.LLM_MODEL,
                input_text=question[:300],
                metadata={"mode": mode, "skill_id": skill_id, "round": round_num},
                trace_id=trace_id,
            )
            try:
                response = await self.openai_client.chat.completions.create(
                    model=settings.LLM_MODEL,
                    messages=current_messages,
                    tools=OPENAI_MATH_TOOLS,
                    tool_choice="auto",
                    temperature=0.3,
                )
            except Exception as e:
                end_generation(agentic_gen, output=f"ERROR: {e}")
                logger.error("AgenticTeacher LLM call failed (round %d): %s", round_num, e)
                raise

            # ── Cost tracking ─────────────────────────────────────────────
            usage = response.usage
            if usage:
                log_call(
                    agent=f"AgenticTeacher-R{round_num}",
                    model=settings.LLM_MODEL,
                    input_tokens=usage.prompt_tokens,
                    output_tokens=usage.completion_tokens,
                    extra=f"mode={mode},skill={skill_id},round={round_num}",
                )

            choice = response.choices[0]
            msg = choice.message

            # ── Final answer: no tool calls ───────────────────────────────
            if not msg.tool_calls:
                end_generation(agentic_gen, output=(msg.content or "")[:500],
                               input_tokens=usage.prompt_tokens if usage else 0,
                               output_tokens=usage.completion_tokens if usage else 0)
                if tool_call_log:
                    logger.info(
                        "🔧 AgenticTeacher: %d tool calls executed for skill=%s",
                        len(tool_call_log), skill_id,
                    )
                return msg.content or ""

            # ── Guard: max rounds reached ─────────────────────────────────
            if round_num >= max_rounds:
                end_generation(agentic_gen, output="[max rounds]",
                               input_tokens=usage.prompt_tokens if usage else 0,
                               output_tokens=usage.completion_tokens if usage else 0)
                logger.warning(
                    "⚠️  AgenticTeacher: max rounds (%d) reached without final answer. "
                    "Returning partial content.", max_rounds,
                )
                return msg.content or "[Không thể hoàn thành sau nhiều bước tính toán]"

            # End this round's generation span (has tool calls)
            end_generation(agentic_gen,
                           output=f"tool_calls: {[tc.function.name for tc in msg.tool_calls]}",
                           input_tokens=usage.prompt_tokens if usage else 0,
                           output_tokens=usage.completion_tokens if usage else 0)

            # ── Append assistant message với tool_calls ────────────────────
            current_messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })

            # ── Execute tools & append results ────────────────────────────
            for tool_call in msg.tool_calls:
                tool_name = tool_call.function.name
                tool_call_id = tool_call.id

                # ── Langfuse span cho mỗi tool call ──
                tool_span = new_span(
                    name=f"tool.{tool_name}",
                    trace_id=trace_id,
                    input_data={
                        "tool": tool_name,
                        "arguments": tool_call.function.arguments[:300],
                    },
                )

                if tool_name not in TOOL_NAMES:
                    tool_result = {"success": False, "error": f"Tool không tồn tại: {tool_name}"}
                    end_span(tool_span, output=tool_result, level="ERROR")
                else:
                    try:
                        kwargs = json.loads(tool_call.function.arguments)
                        tool_fn = math_tools[tool_name]
                        tool_result = tool_fn(**kwargs)
                        end_span(tool_span, output={
                            "success": tool_result.get("success"),
                            "result_latex": tool_result.get("result_latex", "")[:200],
                        })
                    except json.JSONDecodeError as e:
                        tool_result = {"success": False, "error": f"Tham số không hợp lệ: {e}"}
                        end_span(tool_span, output=tool_result, level="ERROR")
                    except Exception as e:
                        tool_result = {"success": False, "error": str(e)}
                        end_span(tool_span, output=tool_result, level="ERROR")

                # Format kết quả cho LLM
                result_text = _format_tool_result(tool_name, tool_result)

                current_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result_text,
                })

                tool_call_log.append({
                    "round": round_num,
                    "tool": tool_name,
                    "args": tool_call.function.arguments,
                    "result": tool_result,
                })

                logger.debug(
                    "🔧 Tool [%s] → %s",
                    tool_name,
                    result_text[:120],
                )

        # Không bao giờ đến đây do guard trên, nhưng để type checker hài lòng
        return "[Tool loop kết thúc không có đáp án]"


# ── Helper functions ───────────────────────────────────────────────────────────

def _normalize_messages(messages: list[Any]) -> list[dict]:
    """
    Chuyển đổi LangChain message objects sang OpenAI dict format.
    Nếu đã là dict rồi thì giữ nguyên.
    """
    result = []
    for m in messages:
        if isinstance(m, dict):
            result.append(m)
        else:
            # LangChain BaseMessage objects
            role_map = {
                "HumanMessage": "user",
                "AIMessage": "assistant",
                "SystemMessage": "system",
            }
            class_name = type(m).__name__
            role = role_map.get(class_name, "user")

            # Handle multimodal content (list of dicts)
            content = m.content
            result.append({"role": role, "content": content})
    return result


def _format_tool_result(tool_name: str, result: dict) -> str:
    """Format tool result thành string để gửi vào tool message."""
    if not result.get("success"):
        return f"❌ Lỗi khi tính {tool_name}: {result.get('error', 'Unknown error')}"

    parts = []

    if "result_latex" in result:
        parts.append(f"Kết quả (LaTeX): {result['result_latex']}")

    if "result_sympy" in result:
        parts.append(f"Kết quả (SymPy): {result['result_sympy']}")

    if "result_float" in result and result["result_float"]:
        parts.append(f"Giá trị số: {result['result_float']}")

    if "solutions" in result:
        solutions_str = ", ".join(result["solutions"]) if result["solutions"] else "∅ (vô nghiệm)"
        parts.append(f"Nghiệm: {solutions_str}")
        parts.append(f"Số nghiệm: {result.get('count', 0)}")

    if "operation" in result:
        parts.append(f"Phép tính: {result['operation']}")

    return "\n".join(parts) if parts else "✅ Tính toán thành công"
