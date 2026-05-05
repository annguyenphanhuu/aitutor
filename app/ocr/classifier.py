"""Question classifier — maps questions to curriculum skills."""

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from app.config import get_settings
from app.knowledge_tracing.skill_graph import SKILLS
import json

settings = get_settings()

CLASSIFIER_PROMPT = """Phân tích câu hỏi Toán 12 sau và xác định:
1. Kỹ năng/chuyên đề liên quan (skill_id)
2. Chương học
3. Độ khó (easy, medium, hard)

Danh sách skill_id:
{skills}

Trả về JSON:
{{"skill_id": "...", "chapter": "...", "difficulty": "easy|medium|hard", "topic_summary": "Mô tả ngắn"}}

CHỈ TRẢ VỀ JSON.
"""


class QuestionClassifier:
    """Classifies math questions into curriculum topics."""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.0,
        )

    async def classify(self, question: str) -> dict:
        """Classify a question into the skill graph."""
        skills_text = "\n".join(
            [f"- {k}: {v['name']} — {v['description']}" for k, v in SKILLS.items()]
        )

        messages = [
            SystemMessage(content=CLASSIFIER_PROMPT.format(skills=skills_text)),
            HumanMessage(content=question),
        ]

        response = await self.llm.ainvoke(messages)

        try:
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                content = content.rsplit("```", 1)[0]
            return json.loads(content)
        except (json.JSONDecodeError, IndexError):
            return {
                "skill_id": None,
                "chapter": "unknown",
                "difficulty": "medium",
                "topic_summary": question[:100],
            }
