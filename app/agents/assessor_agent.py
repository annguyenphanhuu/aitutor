"""Assessor Agent: grades student answers and reports per-skill feedback."""

from __future__ import annotations

import json

from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.agents.context_utils import format_formulas, format_skill_names, normalize_skill_ids
from app.utils.cost_tracker import log_from_response
from app.utils.llm import compatible_temperature

settings = get_settings()


ASSESSOR_SYSTEM_PROMPT = """Ban la giam khao cham bai Toan 12.

Bai toan yeu cau cac ky nang: {skill_names_list}

Cong thuc ap dung:
{formulas_list}

QUY TRINH CHAM (BAT BUOC lam theo thu tu):
1. Xac dinh DE BAI GOC — bai toan hoc sinh dang giai. Neu co lich su hoi thoai,
   de bai goc la bai toan duoc neu ra som nhat; cac tin nhan cham diem/nhan xet
   truoc do ("Diem:", "Nhan xet:", "Loi giai dung:") KHONG phai de bai.
2. TU GIAI de bai goc hoan chinh de co dap an chuan truoc khi cham.
3. Doi chieu TUNG KET LUAN trong cau tra loi MOI NHAT cua hoc sinh voi dap an chuan.
   Dac biet chu y loi DAO NGUOC: cuc dai/cuc tieu, dong bien/nghich bien,
   lon nhat/nho nhat, dau bat dang thuc.
4. Kiem tra hoc sinh co ap dung dung cong thuc khong va co quen dieu kien xac dinh khong.
5. Voi tung ky nang trong danh sach, xac dinh hoc sinh da lam dung hay sai o buoc nao.
6. Cho diem tong the tu 0.0 den 1.0.

RANG BUOC NHAT QUAN (BAT BUOC):
- is_correct = true CHI KHI moi ket luan cua hoc sinh deu dung so voi dap an chuan.
- Neu bat ky ket luan nao sai (ke ca khi cac buoc trung gian dung), is_correct = false,
  score < 1.0 va phan anh phan lam dung.
- feedback, score, is_correct va skills_assessed PHAI thong nhat voi nhau:
  khong duoc vua khen dung vua chi ra loi sai.

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


class AssessorAgent:
    """Agent that evaluates student answers."""

    def __init__(self):
        temp = compatible_temperature(settings.LLM_MODEL_MINI, 0.1)
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
        chat_history: list[dict] | None = None,
    ) -> dict:
        """Evaluate a student's answer with per-skill assessment JSON."""
        skill_ids = normalize_skill_ids(skill_ids)
        system_prompt = ASSESSOR_SYSTEM_PROMPT.format(
            skill_names_list=format_skill_names(skill_ids),
            formulas_list=format_formulas(formula_ids),
        )

        history_block = self._format_history(chat_history)
        prompt = f"""CAU HOI:
{question}
{history_block}
CAU TRA LOI CUA HOC SINH (can cham):
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

        return self._enforce_consistency(result)

    @staticmethod
    def _format_history(chat_history: list[dict] | None) -> str:
        """Render recent turns so the grader can locate the original problem."""
        if not chat_history:
            return "\n"
        lines = []
        for item in chat_history[-6:]:
            role = "Hoc sinh" if item.get("role") == "user" else "Gia su"
            content = str(item.get("content", ""))[:500]
            lines.append(f"{role}: {content}")
        transcript = "\n".join(lines)
        return (
            "\nLICH SU HOI THOAI GAN DAY (chi de xac dinh de bai goc — "
            "cac luot cham diem truoc do KHONG phai de bai):\n"
            f"{transcript}\n\n"
        )

    @staticmethod
    def _enforce_consistency(result: dict) -> dict:
        """Keep is_correct/score coherent with per-skill verdicts."""
        skills = result.get("skills_assessed") or {}
        skill_results = [
            bool(s.get("passed")) for s in skills.values() if isinstance(s, dict)
        ]
        if skill_results and not all(skill_results):
            result["is_correct"] = False

        try:
            score = float(result.get("score", 0.0))
        except (TypeError, ValueError):
            score = 1.0 if result.get("is_correct") else 0.0
        score = min(max(score, 0.0), 1.0)

        if not result.get("is_correct") and score >= 1.0:
            partial = (
                round(sum(skill_results) / len(skill_results), 2)
                if skill_results
                else 0.5
            )
            score = min(partial, 0.9)
        if not result.get("is_correct") and result.get("error_type") == "none":
            result["error_type"] = "conceptual"
        result["score"] = score
        return result
