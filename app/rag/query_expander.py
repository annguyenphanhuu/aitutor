"""
RAG Query Expander — Semantic Routing cho bài Toán lớp 12.

Luồng hoạt động:
  1. Nhận câu hỏi thô (text, có thể cả mô tả hình ảnh)
  2. Gửi cho LLM nhỏ (gpt-4o-mini) cùng taxonomy đầy đủ:
       - Danh sách Chapter có trong Knowledge Base
       - Danh sách Skill ID có trong Knowledge Base
       - Danh sách Formula ID có trong registry
  3. LLM suy luận ra: chapters[], skill_ids[], formula_ids[] phù hợp
  4. Validate kết quả — chỉ giữ những giá trị nằm trong whitelist KB thực tế
  5. Trả về QueryExpansion dataclass để retrieve_with_trace() dùng làm soft-boost

Tại sao cần module này (thay vì dùng metadata đề thi):
  - Metadata đề thi không đầy đủ / không nhất quán
  - Một bài Toán thường cross-chapter (vd. cực trị + hình học)
  - LLM thấy toàn bộ taxonomy → suy luận đúng hơn metadata tĩnh
  - Kết quả được validate → không bao giờ boost nhầm chapter không tồn tại
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Taxonomy không được hard-code ở đây. Skill graph và metadata của các registry
# nội dung là nguồn dữ liệu gốc; nhờ vậy QueryExpander không bị lệch khi KB đổi.
_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@lru_cache(maxsize=1)
def _load_taxonomy() -> tuple[list[str], list[str]]:
    """Load chapter/skill whitelist từ skill graph và dữ liệu RAG thực tế."""
    from app.knowledge_tracing.skill_graph import SKILLS, get_chapters
    from app.rag.formula_registry import list_formulas

    chapters = list(get_chapters())
    skill_ids = list(SKILLS)

    def add_metadata(metadata: object) -> None:
        if not isinstance(metadata, dict):
            return
        chapter = str(metadata.get("chapter", "")).strip()
        skill_id = str(metadata.get("skill_id", "")).strip()
        if chapter and chapter not in chapters:
            chapters.append(chapter)
        if skill_id and skill_id not in skill_ids:
            skill_ids.append(skill_id)

    # Formula registry là nguồn chuẩn cho formulas.json.
    for formula in list_formulas():
        if isinstance(formula, dict):
            add_metadata(formula.get("metadata"))

    # theory.json là source-of-truth trước khi được ingest vào ChromaDB.
    theory_path = _DATA_DIR / "theory.json"
    try:
        raw_theory = json.loads(theory_path.read_text(encoding="utf-8"))
        if isinstance(raw_theory, list):
            for document in raw_theory:
                if isinstance(document, dict):
                    add_metadata(document.get("metadata"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Không thể load taxonomy từ %s: %s", theory_path, exc)

    return chapters, skill_ids


def _normalize_taxonomy_key(value: object) -> str:
    """Normalize whitespace/dash/case để đối chiếu metadata nhất quán."""
    text = str(value).replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip().casefold()

# ── Prompt template ────────────────────────────────────────────────────────────

_EXPAND_PROMPT = """\
Bạn là chuyên gia phân tích bài Toán lớp 12 Việt Nam.

NHIỆM VỤ: Phân tích câu hỏi bên dưới và xác định CHÍNH XÁC những kiến thức cần thiết để giải.

CÂU HỎI:
{question}

DANH SÁCH CHAPTER HỢP LỆ (chỉ chọn từ danh sách này):
{chapters}

DANH SÁCH SKILL_ID HỢP LỆ (chỉ chọn từ danh sách này):
{skill_ids}

DANH SÁCH FORMULA_ID HỢP LỆ (chỉ chọn từ danh sách này):
{formula_ids}

QUY TẮC:
- Chỉ chọn chapter/skill/formula thực sự CẦN THIẾT để giải bài (không chọn thừa)
- Một bài có thể cần nhiều chapter/skill (cross-chapter)
- Nếu bài có đồ thị/hình ảnh → vẫn cần lý thuyết để đọc đồ thị đó
- Chỉ trả về JSON, không giải thích thêm

TRẢ VỀ JSON theo đúng format:
{{
  "chapters": ["<chapter_name>", ...],
  "skill_ids": ["<skill_id>", ...],
  "formula_ids": ["<formula_id>", ...],
  "rag_query": "<câu truy vấn học thuật ngắn gọn để tìm kiếm lý thuyết liên quan, dùng THUẬT NGỮ TOÁN HỌC thay vì ngôn ngữ bài toán. VD: 'xác suất có điều kiện Bayes công thức xác suất toàn phần' thay vì 'kho hàng sản phẩm hỏng'>",
  "reasoning": "<1 câu giải thích ngắn gọn tại sao chọn những mục này>"
}}
"""


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class QueryExpansion:
    """Kết quả phân tích của LLM — đã được validate."""
    chapters:    list[str] = field(default_factory=list)
    skill_ids:   list[str] = field(default_factory=list)
    formula_ids: list[str] = field(default_factory=list)
    # Query Rewriting: LLM viết lại bài toán thực tế → ngôn ngữ học thuật.
    # Dùng làm vector search query thay cho câu hỏi gốc.
    # VD input:  "Một kho hàng 85% loại I, 1% bị hỏng..."
    # VD output: "xác suất có điều kiện Bayes xác suất toàn phần biến cố"
    rag_query:   str = ""     # empty → fallback sang question gốc
    reasoning:   str = ""
    raw_response: str = ""    # để debug
    error:        Optional[str] = None

    @property
    def is_empty(self) -> bool:
        return not self.chapters and not self.skill_ids and not self.formula_ids


# ── Taxonomy loader (formula_ids từ formula_registry) ─────────────────────────

@lru_cache(maxsize=1)
def _load_formula_ids() -> list[str]:
    """Load formula IDs từ registry (cached)."""
    try:
        from app.rag.formula_registry import list_formula_ids
        return list_formula_ids()
    except Exception:
        return []


# ── Core expander ──────────────────────────────────────────────────────────────

class QueryExpander:
    """
    LLM-based semantic router cho RAG pipeline.

    Parameters
    ----------
    model : str
        Model nhỏ dùng làm analyzer (gpt-4o-mini đủ dùng, không cần mạnh).
    api_key : str
        OpenAI API key.
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: str = ""):
        self.model = model
        self.api_key = api_key
        self.known_chapters, self.known_skill_ids = _load_taxonomy()
        self._known_chapters_lower = {
            _normalize_taxonomy_key(c): c for c in self.known_chapters
        }
        self._known_skills_lower = {
            _normalize_taxonomy_key(s): s for s in self.known_skill_ids
        }

    def _build_prompt(self, question: str) -> str:
        formula_ids = _load_formula_ids()
        return _EXPAND_PROMPT.format(
            question=question[:800],  # truncate dài quá
            chapters="\n".join(f"  - {c}" for c in self.known_chapters),
            skill_ids="\n".join(f"  - {s}" for s in self.known_skill_ids),
            formula_ids="\n".join(f"  - {f}" for f in formula_ids) if formula_ids else "  (chưa có)",
        )

    def _validate(self, raw: dict) -> QueryExpansion:
        """
        Validate output LLM — chỉ giữ values nằm trong whitelist.
        Tránh hallucinate chapter/skill không tồn tại gây boost nhầm.
        """
        def list_field(name: str) -> list:
            value = raw.get(name, [])
            return list(value) if isinstance(value, (list, tuple, set)) else []

        valid_chapters = []
        for c in list_field("chapters"):
            c_lower = _normalize_taxonomy_key(c)
            if not c_lower:
                continue
            # Exact match
            if c_lower in self._known_chapters_lower:
                valid_chapters.append(self._known_chapters_lower[c_lower])
            else:
                # Fuzzy: kiểm tra xem có chapter nào chứa chuỗi này không
                for known_lower, known_orig in self._known_chapters_lower.items():
                    if c_lower in known_lower or known_lower in c_lower:
                        valid_chapters.append(known_orig)
                        break

        valid_skills = []
        for s in list_field("skill_ids"):
            s_lower = _normalize_taxonomy_key(s)
            if s_lower in self._known_skills_lower:
                valid_skills.append(self._known_skills_lower[s_lower])

        formula_whitelist = set(_load_formula_ids())
        valid_formulas = [
            str(f).strip() for f in list_field("formula_ids")
            if str(f).strip() in formula_whitelist
        ]

        # Chỉ chấp nhận một chuỗi truy vấn ngắn. Không biến list/dict do model
        # trả sai schema thành query rác và không để prompt dài lọt vào retrieval.
        raw_rag_query = raw.get("rag_query", "")
        rag_query = ""
        if isinstance(raw_rag_query, str):
            rag_query = re.sub(r"\s+", " ", raw_rag_query).strip()[:400]

        return QueryExpansion(
            chapters=list(dict.fromkeys(valid_chapters)),   # dedup, giữ thứ tự
            skill_ids=list(dict.fromkeys(valid_skills)),
            formula_ids=list(dict.fromkeys(valid_formulas)),
            rag_query=rag_query,
            reasoning=str(raw.get("reasoning", ""))[:500],
        )

    def expand(self, question: str) -> QueryExpansion:
        """
        Phân tích câu hỏi và trả về QueryExpansion đã validate.
        Gọi đồng bộ (sync) — dùng cho retrieve pipeline.
        """
        if not question.strip():
            return QueryExpansion(error="Empty question")

        prompt = self._build_prompt(question)

        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw_text = resp.choices[0].message.content or "{}"
            raw_dict = json.loads(raw_text)
            result = self._validate(raw_dict)
            result.raw_response = raw_text
            logger.debug(
                "QueryExpander: chapters=%s skills=%s formulas=%s reason=%s",
                result.chapters, result.skill_ids, result.formula_ids, result.reasoning,
            )
            return result

        except json.JSONDecodeError as e:
            logger.warning("QueryExpander JSON parse error: %s", e)
            return QueryExpansion(error=f"JSON parse error: {e}")
        except Exception as e:
            logger.warning("QueryExpander LLM call failed: %s", e)
            return QueryExpansion(error=str(e))


# ── Singleton helper ───────────────────────────────────────────────────────────

_expander: Optional[QueryExpander] = None


def get_query_expander() -> QueryExpander:
    """Trả về singleton QueryExpander (lazy init với settings)."""
    global _expander
    if _expander is None:
        from app.config import get_settings
        s = get_settings()
        _expander = QueryExpander(
            model=s.LLM_MODEL_NANO,
            api_key=s.OPENAI_API_KEY,
        )
    return _expander


def expand_question(question: str) -> QueryExpansion:
    """Shortcut: expand câu hỏi với singleton expander."""
    return get_query_expander().expand(question)
