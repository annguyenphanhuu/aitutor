"""
ExamSolver — End-to-end exam solving pipeline.

Pipeline:
  1. Upload → OCR (Cloud VLM or Local GOT-OCR2.0)
  2. Question Splitting (LLM)
  3. Per-Question Solving (semi-parallel, Semaphore=5):
     - Classify skill (reuse Orchestrator logic)
     - GraphRAG retrieval (per question — ensures accurate context)
     - Teacher Agent (answer mode — direct solution)
  4. Aggregation → formatted Markdown + LaTeX report

KEY DESIGN DECISIONS:
  • Dual OCR Engine (Strategy Pattern): swap cloud/local without pipeline change
  • Per-Question Solving: each question gets its own RAG retrieval → accurate context
  • Semi-Parallel (Semaphore): 5x faster than sequential, same accuracy
  • asyncio.gather() preserves question order in output
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.ocr.ocr_strategy import get_ocr_engine
from app.ocr.question_splitter import QuestionSplitter
from app.agents.teacher_agent import TeacherAgent
from app.knowledge_tracing.service import get_all_masteries
from app.knowledge_tracing.skill_graph import SKILLS

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class QuestionSolution:
    """Solution for a single exam question."""
    question_number: str
    question_type: str
    content: str          # original question text
    skill_id: Optional[str] = None
    skill_name: Optional[str] = None
    solution: str = ""
    error: Optional[str] = None


@dataclass
class ExamSolverResult:
    """Aggregated result from the exam solver pipeline."""
    total_questions: int = 0
    solutions: list[QuestionSolution] = field(default_factory=list)
    report_markdown: str = ""
    skill_stats: dict = field(default_factory=dict)
    raw_ocr: str = ""
    ocr_engine_used: str = "cloud"
    elapsed_seconds: float = 0.0
    extracted_images: list = field(default_factory=list)  # ExtractedImage objects


# Type for progress callback: (current, total, question_number) → None
ProgressCallback = Callable[[int, int, str], None]


class ExamSolver:
    """Orchestrates the full exam solving pipeline.

    Parameters
    ----------
    ocr_engine : str
        "cloud" → GPT-4o-mini Vision | "local" → GOT-OCR2.0
    """

    def __init__(self, ocr_engine: str = "cloud"):
        self.ocr = get_ocr_engine(ocr_engine)
        self.ocr_engine_name = ocr_engine
        self.splitter = QuestionSplitter()
        self.teacher = TeacherAgent(model=settings.EXAM_SOLVER_MODEL)
        self._semaphore = asyncio.Semaphore(settings.SOLVE_CONCURRENCY)

        # Classifier — reuse Orchestrator's OpenAI client
        from openai import AsyncOpenAI
        self._classifier_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async def solve(
        self,
        file_bytes: bytes,
        file_type: str,       # "image" | "pdf"
        mime_type: str,
        user_id: int,
        db: AsyncSession,
        on_progress: Optional[ProgressCallback] = None,
        raw_ocr_text: Optional[str] = None,
    ) -> ExamSolverResult:
        """Full pipeline: OCR → Split → Solve each → Aggregate.

        Parameters
        ----------
        file_bytes : bytes
            Raw file content (image or PDF).
        file_type : str
            "image" or "pdf".
        mime_type : str
            MIME type of the uploaded file.
        user_id : int
            Current user ID for mastery lookup.
        db : AsyncSession
            Database session.
        on_progress : callable, optional
            Async callback (current, total, question_number) for SSE updates.
        raw_ocr_text : str, optional
            If provided, skips the OCR step and uses this text directly.

        Returns
        -------
        ExamSolverResult
        """
        t0 = time.monotonic()

        # ── Step 1: OCR ─────────────────────────────────────────
        if raw_ocr_text:
            logger.info("🔍 Exam Solver: Using provided OCR text (engine=%s, type=%s)", self.ocr_engine_name, file_type)
            raw_ocr = raw_ocr_text
        else:
            logger.info("🔍 Exam Solver: OCR start (engine=%s, type=%s)", self.ocr_engine_name, file_type)

        extracted_images = []
        page_images: dict[int, bytes] = {}  # page_num (1-indexed) → PNG bytes

        if file_type == "pdf":
            if not raw_ocr_text:
                page_texts = await self.ocr.ocr_pdf(file_bytes)
                raw_ocr = "\n\n---\n\n".join(
                    f"[Trang {i+1}]\n{text}" for i, text in enumerate(page_texts)
                )
                n_ocr_pages = len(page_texts)
            else:
                n_ocr_pages = 50 # Fallback: render up to 50 pages if raw_ocr_text is provided

            # Step 1b: Render page images for Vision-based solving
            try:
                import fitz
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                n_render_pages = min(len(doc), n_ocr_pages)
                for pg_num in range(n_render_pages):
                    page = doc[pg_num]
                    pix = page.get_pixmap(dpi=200)  # 200 DPI balances quality vs token cost
                    page_images[pg_num + 1] = pix.tobytes("png")
                doc.close()
                logger.info("📸 Rendered %d page images for Vision", len(page_images))
            except Exception as e:
                logger.warning("⚠️ Page image rendering failed: %s", e)

            # Step 1c: Extract standalone images/graphs
            try:
                from app.ocr.image_extractor import extract_images_from_pdf
                extracted_images = extract_images_from_pdf(file_bytes)
                logger.info("📸 Extracted %d standalone images from PDF", len(extracted_images))
            except Exception as e:
                logger.warning("⚠️ Image extraction failed: %s", e)
        else:
            if not raw_ocr_text:
                raw_ocr = await self.ocr.ocr_image(file_bytes, mime_type)
            # For single images, store as page 1
            page_images[1] = file_bytes

        logger.info("📄 OCR done: %d chars", len(raw_ocr))

        # ── Step 2: Split questions ──────────────────────────────
        questions = await self.splitter.split(raw_ocr)
        total = len(questions)
        n_with_fig = sum(1 for q in questions if q.get("has_figure"))
        logger.info("📋 Split done: %d questions (%d with figures)", total, n_with_fig)

        if total == 0:
            return ExamSolverResult(
                raw_ocr=raw_ocr,
                ocr_engine_used=self.ocr_engine_name,
                report_markdown="⚠️ Không tách được câu hỏi nào từ đề thi. Hãy kiểm tra lại file.",
                elapsed_seconds=time.monotonic() - t0,
            )

        # ── Step 3: Solve semi-parallel (Semaphore) ──────────────
        # Each question is fully independent → safe to parallelize
        masteries = await get_all_masteries(db, user_id)
        progress_counter = {"done": 0}

        async def _solve_with_semaphore(q: dict) -> QuestionSolution:
            async with self._semaphore:
                result = await self._solve_single(q, masteries, page_images)
                progress_counter["done"] += 1
                if on_progress:
                    on_progress(
                        progress_counter["done"],
                        total,
                        q["question_number"],
                    )
                return result

        solutions: list[QuestionSolution] = list(
            await asyncio.gather(
                *[_solve_with_semaphore(q) for q in questions]
            )
        )

        elapsed = time.monotonic() - t0
        logger.info(
            "✅ Exam Solver done: %d questions in %.1fs (engine=%s)",
            total, elapsed, self.ocr_engine_name,
        )

        # ── Step 4: Aggregate ────────────────────────────────────
        return self._aggregate(solutions, raw_ocr, elapsed, extracted_images)

    async def _solve_single(
        self,
        question: dict,
        masteries: dict[str, float],
        page_images: dict[int, bytes] | None = None,
    ) -> QuestionSolution:
        """Solve a single question: classify → RAG → Teacher.

        If the question has_figure and page images are available,
        uses Vision model (respond_with_image) for accurate visual reasoning.
        Otherwise uses text-only model (respond) which is cheaper.
        """
        content = question["content"]
        q_num = question["question_number"]
        has_figure = question.get("has_figure", False)
        figure_pages = question.get("figure_pages", [])

        try:
            # Classify skill
            classification = await self._classify_question(content)
            skill_id = classification.get("skill_id")
            skill_ids = classification.get("skill_ids", [])
            formula_ids = classification.get("formula_ids", [])

            if not skill_ids and skill_id:
                skill_ids = [skill_id]
            if not skill_id and skill_ids:
                skill_id = skill_ids[0]

            # Route: Vision (has_figure) or Text-only
            if has_figure and page_images and figure_pages:
                # Get the first relevant page image
                img_bytes = None
                for pg in figure_pages:
                    if pg in page_images:
                        img_bytes = page_images[pg]
                        break
                # Fallback: try all pages if specific page not found
                if not img_bytes and page_images:
                    img_bytes = next(iter(page_images.values()))

                if img_bytes:
                    logger.info(
                        "🖼️ %s: Vision solve (figure on page %s)",
                        q_num, figure_pages,
                    )
                    response = await self.teacher.respond_with_image(
                        image_bytes=img_bytes,
                        image_mime="image/png",
                        ocr_text=content,
                        user_text=f"Giải {q_num}. Chỉ giải câu này, không giải câu khác.",
                        mode="answer",
                        mastery_level="proficient",
                        skill_id=skill_id,
                        skill_ids=skill_ids,
                        formula_ids=formula_ids,
                        masteries=masteries,
                        p_mastery=masteries.get(skill_id, 0.5) if skill_id else 0.5,
                    )
                else:
                    # No image available, fallback to text
                    response = await self.teacher.respond(
                        question=content,
                        mode="answer",
                        mastery_level="proficient",
                        skill_id=skill_id,
                        skill_ids=skill_ids,
                        formula_ids=formula_ids,
                        masteries=masteries,
                        p_mastery=masteries.get(skill_id, 0.5) if skill_id else 0.5,
                    )
            else:
                # Text-only solve (cheaper, faster)
                response = await self.teacher.respond(
                    question=content,
                    mode="answer",
                    mastery_level="proficient",
                    skill_id=skill_id,
                    skill_ids=skill_ids,
                    formula_ids=formula_ids,
                    masteries=masteries,
                    p_mastery=masteries.get(skill_id, 0.5) if skill_id else 0.5,
                )

            skill_info = SKILLS.get(skill_id, {}) if skill_id else {}
            return QuestionSolution(
                question_number=q_num,
                question_type=question["question_type"],
                content=content,
                skill_id=skill_id,
                skill_name=skill_info.get("name"),
                solution=response,
            )

        except Exception as e:
            logger.error("❌ Failed to solve %s: %s", q_num, e)
            return QuestionSolution(
                question_number=q_num,
                question_type=question["question_type"],
                content=content,
                solution=f"⚠️ Không thể giải câu này. Lỗi: {str(e)}",
                error=str(e),
            )

    async def _classify_question(self, content: str) -> dict:
        """Lightweight classification — extract skill_id for RAG routing."""
        from app.knowledge_tracing.skill_graph import SKILLS as _SKILLS
        from app.rag.formula_registry import list_formulas

        skills_list = "\n".join(
            f"- {k}: {v['name']} ({v['chapter']})" for k, v in _SKILLS.items()
        )
        try:
            formulas_list = "\n".join(
                f"- {f['id']}: {f.get('metadata', {}).get('chapter', '')}"
                for f in list_formulas()
            )
        except Exception:
            formulas_list = ""

        prompt = f"""Phân loại câu hỏi Toán 12 sau. Xác định skill_id chính và formula_ids liên quan.

Danh sách skill_id:
{skills_list}

Danh sách formula_id:
{formulas_list}

Trả về JSON:
{{"skill_id": "skill_id_chinh_hoac_null", "skill_ids": ["skill_id_1"], "formula_ids": ["formula_id_1"]}}

CHỈ TRẢ VỀ JSON."""

        try:
            response = await self._classifier_client.responses.create(
                model=settings.LLM_MODEL_MINI,
                input=[
                    {"role": "user", "content": f"{prompt}\n\nCâu hỏi: {content[:1500]}"},
                ],
            )

            text = response.output_text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                text = text.rsplit("```", 1)[0]
            result = json.loads(text)

            # Cost tracking
            from app.utils.cost_tracker import log_call
            input_tokens = getattr(response.usage, "input_tokens", 0) if response.usage else 0
            output_tokens = getattr(response.usage, "output_tokens", 0) if response.usage else 0
            log_call(
                agent="ExamSolver.Classifier",
                model=settings.LLM_MODEL_MINI,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

            return result
        except Exception:
            return {"skill_id": None, "skill_ids": [], "formula_ids": []}

    def _aggregate(
        self,
        solutions: list[QuestionSolution],
        raw_ocr: str,
        elapsed: float,
        extracted_images: list | None = None,
    ) -> ExamSolverResult:
        """Aggregate per-question solutions into a formatted report."""

        # Skill statistics
        skill_stats: dict[str, int] = {}
        for s in solutions:
            if s.skill_id:
                skill_stats[s.skill_id] = skill_stats.get(s.skill_id, 0) + 1

        # Build Markdown report
        parts: list[str] = []
        for s in solutions:
            skill_label = f" `[{s.skill_name}]`" if s.skill_name else ""
            section = f"### {s.question_number}{skill_label}\n\n"
            section += f"**Đề bài:**\n{s.content}\n\n"
            section += f"**Lời giải:**\n{s.solution}\n\n"
            section += "---\n\n"
            parts.append(section)

        n_errors = sum(1 for s in solutions if s.error)
        total = len(solutions)

        report = f"# 📝 Lời Giải Đề Thi\n\n"
        report += f"**Tổng số câu:** {total}"
        if n_errors:
            report += f" (⚠️ {n_errors} câu gặp lỗi)"
        report += f" | **Thời gian:** {elapsed:.1f}s\n\n"
        report += "".join(parts)

        return ExamSolverResult(
            total_questions=total,
            solutions=solutions,
            report_markdown=report,
            skill_stats=skill_stats,
            raw_ocr=raw_ocr,
            ocr_engine_used=self.ocr_engine_name if hasattr(self, 'ocr_engine_name') else "cloud",
            elapsed_seconds=elapsed,
            extracted_images=extracted_images or [],
        )
