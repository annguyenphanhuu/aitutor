"""Planner Agent — creates personalized study plans based on mastery profile."""

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from app.config import get_settings
from app.knowledge_tracing.skill_graph import SKILLS
import json

settings = get_settings()


PLANNER_SYSTEM_PROMPT = """Bạn là chuyên gia lập kế hoạch học tập Toán 12.
Dựa vào hồ sơ năng lực của học sinh, hãy đề xuất kế hoạch ôn tập.

QUY TẮC:
1. Ưu tiên kỹ năng nền tảng (prerequisite) trước kỹ năng nâng cao.
2. Mỗi khuyến nghị cần cụ thể: số lượng bài tập, chuyên đề, thời gian.
3. Tập trung vào các kỹ năng có mastery < 0.6 (chưa thành thạo).
4. Đề xuất tối đa 5 mục ưu tiên.

BẮT BUỘC trả về JSON theo format:
{{
    "summary": "Tóm tắt ngắn gọn tình trạng học tập",
    "priorities": [
        {{
            "skill_id": "id_kỹ_năng",
            "skill_name": "Tên kỹ năng",
            "action": "Mô tả cụ thể cần làm",
            "exercises": 5,
            "urgency": "high" | "medium" | "low"
        }}
    ],
    "encouragement": "Lời động viên cho học sinh"
}}

CHỈ TRẢ VỀ JSON, KHÔNG CÓ TEXT KHÁC.
"""


class PlannerAgent:
    """Agent that generates personalized study recommendations."""

    def __init__(self):
        # MINI: structured study plan from mastery data, no deep reasoning
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL_MINI,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.4,
        )

    async def create_plan(self, mastery_profile: list[dict]) -> dict:
        """
        Generate a study plan based on student mastery profile.

        mastery_profile: list of {skill_id, skill_name, p_mastery, level, ...}
        """
        # Format profile for the LLM
        profile_text = "HỒ SƠ NĂNG LỰC HỌC SINH:\n"
        for m in mastery_profile:
            profile_text += (
                f"- {m['skill_name']} ({m['skill_id']}): "
                f"mastery={m['p_mastery']}, level={m['level']}, "
                f"attempts={m.get('total_attempts', 0)}\n"
            )

        if not mastery_profile:
            profile_text += "Chưa có dữ liệu. Đây là học sinh mới.\n"

        # Add skill prerequisites info
        profile_text += "\nDANH SÁCH KỸ NĂNG VÀ PREREQUISITES:\n"
        for skill_id, info in SKILLS.items():
            prereqs = info["prerequisites"]
            prereq_str = ", ".join(prereqs) if prereqs else "Không có"
            profile_text += f"- {info['name']} ({skill_id}): prerequisites=[{prereq_str}]\n"

        messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=profile_text),
        ]

        response = await self.llm.ainvoke(messages)

        try:
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                content = content.rsplit("```", 1)[0]
            result = json.loads(content)
        except (json.JSONDecodeError, IndexError):
            result = {
                "summary": "Không thể tạo kế hoạch tự động.",
                "priorities": [],
                "encouragement": "Hãy tiếp tục luyện tập!",
            }

        return result
