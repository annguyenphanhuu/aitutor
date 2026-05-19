"""Assessor Agent: grades student answers and reports per-skill feedback."""

from __future__ import annotations

import json

from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.knowledge_tracing.skill_graph import SKILLS
from app.rag.formula_registry import get_formulas_by_ids
from app.utils.cost_tracker import log_from_response

settings = get_settings()


ASSESSOR_SYSTEM_PROMPT = """Ban la giam khao cham bai Toan 12.

Bai toan yeu cau cac ky nang: {skill_names_list}

Cong thuc ap dung:
{formulas_list}

NHIEM VU:
1. Kiem tra hoc sinh co ap dung dung cong thuc khong va co quen dieu kien xac dinh khong.
2. Voi tung ky nang trong danh sach, xac dinh hoc sinh da lam dung hay sai o buoc nao.
3. Cho diem tong the tu 0.0 den 1.0.

Bat buoc tra ve JSON:
{{
  "is_correct": true/false,
  "score": 0.0-1.0,
  "error_type": "none" | "calculation" | "conceptual" | "procedural",
  "skills_assessed": {{
    "skill_id": {{"passed": true/false, "feedback": "Nhan xet ngan"}}
  }},
  "feedback": "Nhan xet tong the bang tieng Viet",
  "overall_feedback": "Nhan xet tong the bang tieng Viet",
  "correct_solution": "Loi giai dung day du"
}}

CHI TRA VE JSON, KHONG THEM VAN BAN KHAC.
"""


def _normalize_skill_ids(skill_ids: list[str] | None) -> list[str]:
    normalized: list[str] = []
    for skill_id in skill_ids or []:
        if skill_id and skill_id not in normalized:
            normalized.append(skill_id)
    return normalized


def _format_skill_names(skill_ids: list[str]) -> str:
    if not skill_ids:
        return "chua xac dinh"
    return ", ".join(
        f"{SKILLS.get(skill_id, {}).get('name', skill_id)} ({skill_id})"
        for skill_id in skill_ids
    )


def _format_formulas(formula_ids: list[str] | None) -> str:
    formulas = get_formulas_by_ids(formula_ids or [])
    if not formulas:
        return "Khong co cong thuc trong tam duoc gan metadata."
    return "\n\n".join(formula.get("content", "") for formula in formulas)


class AssessorAgent:
    """Agent that evaluates student answers."""

    def __init__(self):
        temp = 1.0 if any(p in settings.LLM_MODEL_MINI for p in ["o1", "o3", "o4"]) else 0.1
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL_MINI,
            api_key=settings.OPENAI_API_KEY,
            temperature=temp,
        )

    async def assess(
        self,
        question: str,
        student_answer: str,
        skill_ids: list[str] | None = None,
        formula_ids: list[str] | None = None,
    ) -> dict:
        """Evaluate a student's answer with per-skill assessment JSON."""
        skill_ids = _normalize_skill_ids(skill_ids)
        system_prompt = ASSESSOR_SYSTEM_PROMPT.format(
            skill_names_list=_format_skill_names(skill_ids),
            formulas_list=_format_formulas(formula_ids),
        )

        prompt = f"""CAU HOI:
{question}

CAU TRA LOI CUA HOC SINH:
{student_answer}

Hay cham diem va phan tich."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ]

        response = await self.llm.ainvoke(messages)
        log_from_response(agent="Assessor", model=settings.LLM_MODEL_MINI, response=response)

        try:
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                content = content.rsplit("```", 1)[0]
            result = json.loads(content)
        except (json.JSONDecodeError, IndexError):
            result = {
                "is_correct": False,
                "score": 0.0,
                "error_type": "unknown",
                "skills_assessed": {},
                "feedback": response.content,
                "overall_feedback": response.content,
                "correct_solution": "Khong the phan tich loi giai.",
            }

        if "skills_assessed" not in result:
            result["skills_assessed"] = {
                skill_id: {
                    "passed": bool(result.get("is_correct", False)),
                    "feedback": result.get("feedback", ""),
                }
                for skill_id in skill_ids
            }
        result.setdefault("overall_feedback", result.get("feedback", ""))
        result.setdefault("feedback", result.get("overall_feedback", ""))
        result.setdefault("correct_solution", "")
        result.setdefault("score", 1.0 if result.get("is_correct") else 0.0)
        result.setdefault("error_type", "none" if result.get("is_correct") else "unknown")

        return result
