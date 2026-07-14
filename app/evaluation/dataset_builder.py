"""
Dataset builder for RAGAS evaluation.

Loads questions from exam JSON files and builds a RAGAS EvaluationDataset
by calling the live RAG pipeline + Teacher Agent.

Vision support:
    Questions with has_image=True will have their image base64-encoded
    and passed to a vision-capable LLM (gpt-4o) alongside the question text.
    Text-only questions continue using the standard LLM.

Usage:
    python -m app.evaluation.dataset_builder --limit 10 --output data/ragas_dataset.json
"""

from __future__ import annotations

import asyncio
import json
import logging
import argparse
from pathlib import Path
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from app.config import get_settings
from app.rag.knowledge_base import get_knowledge_base
from app.utils.answer_format import answer_format_instruction, ensure_answer_tag
from app.utils.image_loader import get_image_mime, load_image_base64
from app.utils.llm import compatible_temperature

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_exam_questions(
    exam_dir: str = "data/exams",
    question_types: Optional[list[str]] = None,
    limit: Optional[int] = None,
) -> list[dict]:
    """Load questions from all exam JSON files.

    Returns list of dicts with keys:
        id, question, ground_truth, skill_id, chapter, type,
        correct_answer, has_image, image_path
    """
    question_types = question_types or ["exam_mcq", "exam_short_answer", "exam_true_false"]
    questions = []

    path_obj = Path(exam_dir)
    files_to_process = [path_obj] if path_obj.is_file() else path_obj.glob("*.json")

    for json_file in files_to_process:
        try:
            items = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Cannot load %s: %s", json_file, e)
            continue

        for item in items:
            meta = item.get("metadata", {})
            if meta.get("type") not in question_types:
                continue
            # Skip image-dependent questions that have no answer text at all
            if meta.get("has_image") and not item.get("answer"):
                continue

            questions.append({
                "id":            item.get("id", ""),
                "question":      item.get("content", ""),
                "ground_truth":  item.get("answer", ""),
                "skill_id":      meta.get("skill_id", ""),
                "chapter":       meta.get("chapter", ""),
                "type":          meta.get("type", ""),
                "correct_answer": meta.get("correct_answer", ""),
                "has_image":     meta.get("has_image", False),
                "image_path":    meta.get("image_path", ""),
            })

    if limit:
        questions = questions[:limit]

    logger.info("Loaded %d questions from %s", len(questions), exam_dir)
    return questions


def retrieve_contexts(
    question: str,
    k: int = 5,
    skill_id: Optional[str] = None,
    chapter: Optional[str] = None,
) -> list[str]:
    """Retrieve context chunks for a question using the live RAG pipeline.

    Uses QueryExpander (LLM) to infer relevant chapters/skills from the
    question text, then applies a light soft boost. Metadata from exam JSON
    (skill_id, chapter) is NOT used for retrieval — only for evaluation.
    """
    try:
        kb = get_knowledge_base()

        from app.rag.reranker import get_reranker
        from app.rag.query_expander import expand_question
        from app.rag.graph_rag import build_search_queries, merge_search_results

        reranker = get_reranker()
        pool_k = settings.RERANKER_CANDIDATE_K if settings.RERANKER_ENABLED else k * 2

        # Step 1: Query Expansion
        expansion = expand_question(question)

        # Step 2: Multi-query hybrid search over the theory collection.
        rewritten_query = expansion.rag_query if not expansion.error else ""
        search_queries = build_search_queries(question, rewritten_query)
        candidates = merge_search_results([
            kb.search_theory(search_query, k=pool_k, alpha=0.55)
            for search_query in search_queries
        ])

        if not expansion.error and expansion.formula_ids:
            known_ids = {str(candidate.get("id", "")) for candidate in candidates}
            for formula in kb.get_formulas_by_ids(expansion.formula_ids):
                if str(formula.get("id", "")) not in known_ids:
                    candidates.append({
                        "id": formula["id"],
                        "content": formula["content"],
                        "metadata": formula.get("metadata", {}),
                        "hybrid_score": 1.0,
                    })

        # Step 3: Light soft boost from expansion
        SKILL_BOOST   = 0.12
        CHAPTER_BOOST = 0.07
        expanded_skills   = {s.lower() for s in expansion.skill_ids}
        def normalize_chapter(value: str) -> str:
            return value.replace("–", "-").replace("—", "-").lower().strip()
        expanded_chapters = {normalize_chapter(c) for c in expansion.chapters}

        for r in candidates:
            bonus = 0.0
            meta  = r.get("metadata", {})
            chunk_skill   = str(meta.get("skill_id", "")).lower().strip()
            chunk_chapter = normalize_chapter(str(meta.get("chapter", "")))

            if chunk_skill and chunk_skill in expanded_skills:
                bonus += SKILL_BOOST
            if chunk_chapter and any(
                chunk_chapter in exp_ch or exp_ch in chunk_chapter
                for exp_ch in expanded_chapters
            ):
                bonus += CHAPTER_BOOST

            if bonus > 0:
                r["hybrid_score"] = min(r.get("hybrid_score", 0) + bonus, 1.0)

        # Step 4: Rerank
        ranked = reranker.rerank(rewritten_query or question, candidates, k=k)
        return [r["content"] for r in ranked]
    except Exception as e:
        logger.warning("RAG retrieval failed for question: %s", e)
        return []


async def generate_answer(
    question: str,
    context: str,
    question_type: str = "",
    model: Optional[str] = None,
    image_path: Optional[str] = None,
) -> str:
    """Generate an answer using the Teacher Agent (simplified, no BKT/session).

    Parameters
    ----------
    question : str
        The full question text (LaTeX included).
    context : str
        RAG-retrieved context, pre-joined as a single string.
    question_type : str
        One of exam_mcq / exam_true_false / exam_short_answer.
    model : str, optional
        Override the LLM model name.  When an image is provided and this is
        None, we automatically switch to settings.VISION_LLM_MODEL (default: gpt-5.4).
    image_path : str, optional
        Relative path to an image file (e.g. "data/exams/images/…/q01.png").
        When provided and the file exists, the question is sent as a
        multimodal message (text + image) to a vision-capable LLM.
    """
    # ── Resolve image (if any) ────────────────────────────────────────────────
    b64_image: Optional[str] = None
    media_type: str = "image/png"
    using_vision: bool = False

    if image_path:
        b64_image = load_image_base64(image_path)
        if b64_image:
            media_type = get_image_mime(image_path)
            using_vision = True
            logger.info("  -> Vision mode: %s (%s)", Path(image_path).name, media_type)

    # ── Choose model ─────────────────────────────────────────────────────────
    chosen_model = model or (
        settings.VISION_LLM_MODEL if using_vision else settings.LLM_MODEL
    )

    llm = ChatOpenAI(
        model=chosen_model,
        api_key=settings.OPENAI_API_KEY,
        temperature=compatible_temperature(chosen_model, 0.1),
    )

    # ── Build type-specific format instruction ────────────────────────────────
    format_instruction = answer_format_instruction(question_type)
    if format_instruction:
        format_instruction = "\n\n" + format_instruction

    if question_type == "exam_true_false":
        format_instruction += (
            "\n\nDAC BIET KHI CO HINH VE (Visual Feature Extraction — BAT BUOC):"
            "\nNeu menh de yeu cau danh gia mot do thi / bang bien thien / hinh ve:"
            "\n  [TRICH XUAT THI GIAC] Truoc khi ket luan Dung/Sai, ban PHAI liet ke CU THE:"
            "\n  1. Phuong trinh duong tiem can (ngang va dung) doc tu hinh."
            "\n  2. Toa do cac diem dac trung: cuc dai, cuc tieu, giao diem truc hoanh, y-intercept."
            "\n  3. Chieu bien thien (ham so tang/giam tren tung khoang)."
            "\n  4. Dang bieu do (hyperbol? parabol? ham bac ba?) va tinh doi xung."
            "\n  KHONG DUOC dung phat bieu 'Neu hinh khong the hien dung thi sai' — DO LA"
            "\n  TRANH TRANH LUAN. Ban PHAI doc hinh va ket luan CHINH XAC."
        )
    elif question_type == "exam_short_answer":
        format_instruction += (
            "\nNeu ket qua la xac suat hoac phan tram, ghi duoi dang SO THAP PHAN (vi du: 0.56),"
            " KHONG ghi duoi dang phan tram (vi du: KHONG ghi 56)."
        )

    # ── CoT instruction — vision-aware ────────────────────────────────────────
    cot_instruction = (
        "\n\nQUY TRINH SUY LUAN BAT BUOC — Agentic Tool-Calling:"
        "\n⚠️  NGHIEM CAM: TUYET DOI KHONG duoc tu tinh nham bat ky phep toan nao."
        "\nVoi MOI phep tinh, ban BAT BUOC phai goi Tool (Cong cu SymPy) de may tinh xu ly."
        "\nCac truong hop BAT BUOC phai goi Tool:"
        "\n  - Dao ham: goi compute_derivative"
        "\n  - Tich phan: goi compute_integral"
        "\n  - Giai phuong trinh: goi solve_equation"
        "\n  - Giai bat phuong trinh (>, <, >=, <=): goi solve_inequality"
        "\n  - Tinh logarit, mu, can bac n phuc tap: goi solve_equation hoac simplify_expression"
        "\n  - Tinh gia tri bieu thuc: goi simplify_expression hoac evaluate_at_point"
        "\nQuy trinh:"
        "\n  Buoc 1: Xac dinh dang bai va phuong phap giai."
        "\n  Buoc 2: Thiet lap bieu thuc / phuong trinh / bat phuong trinh can tinh."
        "\n  Buoc 3: GOI TOOL de tinh. Doi ket qua chinh xac tu SymPy roi moi ket luan."
        "\n  Buoc 4: Neu bai co HINH VE hoac DO THI — doc CAN THAN toa do tu hinh:"
        "\n          a) Doc CHINH XAC vi tri tuong doi cua cac diem/duong co nhan (tren/duoi, trai/phai)."
        "\n          b) Ghi lai it nhat 3 cap (x, y) CU THE doc tu hinh."
        "\n          c) Doi chieu TUNG phuong an voi cac diem da doc."
        "\n          KHONG BAO GIO ket luan chi dua vao cam giac hinh dang chung chung."
        "\n  Buoc 5: Bai toan GTLN/GTNN/Toi uu hoa:"
        "\n          a) Thiet lap ham f(t), goi compute_derivative de tinh f'(t)."
        "\n          b) Goi solve_equation de giai f'(t)=0 tim nghiem t*."
        "\n          c) Goi evaluate_at_point de tinh f(t*) chinh xac."
        "\n  Buoc 6: Bai toan tang truong/phan bao/lai kep:"
        "\n          a) Thiet lap bat phuong trinh, goi solve_inequality."
        "\n          b) Doc ket qua bien thap phan tu SymPy."
        "\n          c) Lam tron LEN den boi so nguyen cua chu ky (neu co chu ky roi rac)."
        "\n  Buoc 7: Kiem tra lai dap an bang thu nguoc hoac dieu kien bien."
        "\n  Buoc 8: QUY UOC LOGARIT KHI GOI SYMPY — BAT BUOC THUC HIEN:"
        "\n          Trong toan pho thong Viet Nam, ky hieu 'log' hoac 'lg' MAC DINH la logarit co so 10."
        "\n          Trong SymPy, ham log() MAC DINH la logarit tu nhien (co so e = ln)."
        "\n          ==> De tranh sai, BAT BUOC mapping nhu sau khi goi SymPy:"
        "\n              - 'log(x)'  trong de bai   ==> goi SymPy: log(x, 10)"
        "\n              - 'lg(x)'   trong de bai   ==> goi SymPy: log(x, 10)"
        "\n              - 'ln(x)'   trong de bai   ==> goi SymPy: log(x)       [giu nguyen]"
    )

    # ── RAG-as-reference framing ──────────────────────────────────────────────
    rag_framing = (
        "\n\nCACH SU DUNG TAI LIEU THAM KHAO:"
        "\nPhan tai lieu ben duoi duoc truy xuat TU DONG tu co so du lieu (RAG)"
        "\ndua tren noi dung cau hoi. Noi dung tai lieu la CHINH XAC (trich tu SGK"
        "\nva de thi da kiem duyet), tuy nhien co the KHONG LIEN QUAN truc tiep"
        "\nden cau hoi hien tai. Ban PHAI:"
        "\n  - Danh gia xem tai lieu co thuc su lien quan den cau hoi khong truoc khi dung."
        "\n  - Neu tai lieu lien quan -> tan dung de ho tro lap luan va giai thich."
        "\n  - Neu tai lieu khong lien quan -> bo qua va dung kien thuc Toan noi tai de tra loi."
        "\n  - Luon uu tien suy luan Toan hoc dung, tai lieu chi la bo tro."
    )

    system_text = (
        "Ban la gia su Toan 12. Hay tra loi cau hoi ngan gon va chinh xac."
        + format_instruction
        + cot_instruction
        + rag_framing
        + "\n\nTAI LIEU THAM KHAO (noi dung chinh xac, muc do lien quan can danh gia):\n" + context
    )

    # ── Build messages ────────────────────────────────────────────────────────
    if using_vision and b64_image:
        # Multimodal: text + image
        human_content = [
            {"type": "text", "text": question},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{b64_image}",
                    "detail": "high",
                },
            },
        ]
        messages = [
            SystemMessage(content=system_text),
            HumanMessage(content=human_content),
        ]
    else:
        # Text-only
        messages = [
            SystemMessage(content=system_text),
            HumanMessage(content=question),
        ]

    response = await llm.ainvoke(messages)
    return ensure_answer_tag(response.content, question_type)


async def build_ragas_samples(
    questions: list[dict],
    k: int = 5,
) -> list[dict]:
    """
    Build RAGAS-compatible samples from questions.

    For image questions (has_image=True), the actual image is encoded and
    passed to a vision LLM for answer generation.  Retrieval is always
    text-only so the context quality can still be measured.

    Returns list of dicts with keys:
        user_input, retrieved_contexts, response, reference
    """
    samples = []

    for i, q in enumerate(questions):
        logger.info("[%d/%d] Processing: %s", i + 1, len(questions), q["id"])

        # Step 1: Retrieve contexts — text-only (image not used here)
        contexts = retrieve_contexts(
            q["question"],
            k=k,
            skill_id=q.get("skill_id"),
            chapter=q.get("chapter"),
        )

        # Step 2: Generate answer
        #   - image questions  → Vision LLM (text + real image)
        #   - text questions   → standard LLM
        context_text = "\n\n---\n\n".join(contexts) if contexts else "Khong tim thay tai lieu."
        image_path = q.get("image_path", "") if q.get("has_image") else ""

        answer = await generate_answer(
            q["question"],
            context_text,
            question_type=q.get("type", ""),
            image_path=image_path,
        )

        samples.append({
            # RAGAS standard fields
            "user_input":         q["question"],
            "retrieved_contexts": contexts,
            "response":           answer,
            "reference":          q["ground_truth"],
            # Extra metadata for analysis
            "_meta": {
                "id":            q["id"],
                "skill_id":      q["skill_id"],
                "chapter":       q["chapter"],
                "type":          q["type"],
                "correct_answer": q["correct_answer"],
                "has_image":     q.get("has_image", False),
                "image_path":    image_path,
            },
        })

    return samples


def save_samples(samples: list[dict], output_path: str) -> None:
    """Save samples to a JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(
        json.dumps(samples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved %d samples to %s", len(samples), output_path)


def load_samples(path: str) -> list[dict]:
    """Load previously saved samples."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ── CLI ───────────────────────────────────────────────────────────────────────

async def _main(args):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    questions = load_exam_questions(
        exam_dir=args.exam_dir,
        limit=args.limit,
    )
    if not questions:
        print("No questions found. Check exam_dir path.")
        return

    samples = await build_ragas_samples(questions, k=args.k)
    save_samples(samples, args.output)
    print(f"Built {len(samples)} RAGAS samples -> {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build RAGAS evaluation dataset")
    parser.add_argument("--exam-dir",  default="data/exams",            help="Path to exam JSON dir")
    parser.add_argument("--output",    default="data/ragas_dataset.json", help="Output JSON path")
    parser.add_argument("--limit",     type=int, default=20,             help="Max questions to process")
    parser.add_argument("--k",         type=int, default=5,              help="Number of RAG chunks to retrieve")
    args = parser.parse_args()

    asyncio.run(_main(args))
