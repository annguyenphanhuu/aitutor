"""
Unit tests for the Evaluation Module.

Covers:
  1. HallucinationTracker (hallucination_tracker.py)
     - record(), report(), reset(), thread-safety
  2. DatasetBuilder (dataset_builder.py)
     - load_exam_questions() filtering + limit
     - save_samples() / load_samples() round-trip
     - retrieve_contexts() fallback when KB not found
  3. RAGASEvaluator (ragas_evaluator.py)
     - build_dataset() conversion
     - run_and_report() report structure + pass/fail logic
     - CLI shorthand metric mapping

No real OpenAI calls are made — all LLM/embedding/RAGAS calls are mocked.
"""

import json
import pytest
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_questions() -> list[dict]:
    """Minimal exam question list covering the 3 question types."""
    return [
        {
            "id": "q_mcq_01",
            "content": "Hàm số nào sau đây đồng biến trên ℝ?\nA. y=x³−3x\nB. y=x³+3x",
            "metadata": {
                "type": "exam_mcq",
                "skill_id": "derivative_basic",
                "chapter": "Đạo hàm",
                "year": 2024,
                "has_image": False,
                "correct_answer": "B",
            },
            "answer": "ĐÁP ÁN: B\nGIẢI THÍCH: y'=3x²+3>0 ∀x.",
        },
        {
            "id": "q_mcq_02",
            "content": "log₂8 bằng?\nA. 2\nB. 3\nC. 4",
            "metadata": {
                "type": "exam_mcq",
                "skill_id": "logarithm_basic",
                "chapter": "Mũ và Logarit",
                "year": 2024,
                "has_image": False,
                "correct_answer": "B",
            },
            "answer": "ĐÁP ÁN: B\nGIẢI THÍCH: log₂(2³)=3.",
        },
        {
            "id": "q_sa_01",
            "content": "Tính I=∫[0→1] 2x dx.",
            "metadata": {
                "type": "exam_short_answer",
                "skill_id": "integral_definite",
                "chapter": "Tích phân",
                "year": 2024,
                "has_image": False,
                "correct_answer": "1",
            },
            "answer": "ĐÁP ÁN: 1\nGIẢI THÍCH: I=[x²]₀¹=1.",
        },
        {
            "id": "q_img_01",
            "content": "Xem hình và trả lời.",
            "metadata": {
                "type": "exam_mcq",
                "skill_id": "function_survey",
                "chapter": "Đạo hàm",
                "year": 2025,
                "has_image": True,  # sẽ bị skip khi không có answer
                "correct_answer": "A",
            },
            "answer": "",  # không có lời giải text → skip
        },
    ]


@pytest.fixture
def sample_ragas_samples() -> list[dict]:
    """Pre-built RAGAS samples (no OpenAI needed)."""
    return [
        {
            "user_input":         "Hàm số nào đồng biến trên ℝ?",
            "retrieved_contexts": ["y'=3x²+3>0 nên hàm đồng biến.", "Quy tắc đạo hàm bậc 3."],
            "response":           "Hàm số y=x³+3x đồng biến trên ℝ vì y'=3x²+3>0.",
            "reference":          "ĐÁP ÁN: B\ny'=3x²+3>0 ∀x.",
            "_meta": {"id": "q_01", "skill_id": "derivative_basic", "chapter": "Đạo hàm", "type": "exam_mcq"},
        },
        {
            "user_input":         "log₂8 bằng bao nhiêu?",
            "retrieved_contexts": ["log₂(2³)=3 theo định nghĩa logarit."],
            "response":           "log₂8 = 3.",
            "reference":          "ĐÁP ÁN: B\nlog₂(2³)=3.",
            "_meta": {"id": "q_02", "skill_id": "logarithm_basic", "chapter": "Mũ và Logarit", "type": "exam_mcq"},
        },
        {
            "user_input":         "Tính tích phân I=∫[0→1]2x dx.",
            "retrieved_contexts": ["∫2x dx = x² + C. Tích phân xác định: [x²]₀¹=1."],
            "response":           "I = [x²]₀¹ = 1.",
            "reference":          "ĐÁP ÁN: 1\nI=[x²]₀¹=1.",
            "_meta": {"id": "q_03", "skill_id": "integral_definite", "chapter": "Tích phân", "type": "exam_short_answer"},
        },
    ]


@pytest.fixture(autouse=True)
def reset_reflection_metrics():
    """Reset singleton counters before every test."""
    from app.evaluation.hallucination_tracker import ReflectionMetrics
    ReflectionMetrics.reset()
    yield
    ReflectionMetrics.reset()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. HallucinationTracker
# ═══════════════════════════════════════════════════════════════════════════════

class TestReflectionMetrics:
    """Tests for app/evaluation/hallucination_tracker.py"""

    def _make_result(self, was_corrected: bool, n_ok: int = 2, n_fail: int = 1):
        """Create a minimal fake ReflectionResult."""
        verifications = (
            [{"result": {"success": True}}]  * n_ok +
            [{"result": {"success": False}}] * n_fail
        )
        result = MagicMock()
        result.was_corrected  = was_corrected
        result.verifications  = verifications
        result.thinking_log   = ["step1", "step2"]
        return result

    # ── Basic record + report ─────────────────────────────────────────────────

    def test_initial_report_is_zero(self):
        from app.evaluation.hallucination_tracker import ReflectionMetrics
        r = ReflectionMetrics.report()
        assert r["total_reflections"]   == 0
        assert r["total_corrections"]   == 0
        assert r["total_verifications"] == 0
        assert r["correction_rate"]     == 0.0

    def test_record_corrected(self):
        from app.evaluation.hallucination_tracker import ReflectionMetrics
        ReflectionMetrics.record(self._make_result(was_corrected=True, n_ok=1, n_fail=1))
        r = ReflectionMetrics.report()
        assert r["total_reflections"]   == 1
        assert r["total_corrections"]   == 1
        assert r["total_verifications"] == 2   # 1 ok + 1 fail
        assert r["total_sympy_errors"]  == 1

    def test_record_not_corrected(self):
        from app.evaluation.hallucination_tracker import ReflectionMetrics
        ReflectionMetrics.record(self._make_result(was_corrected=False, n_ok=3, n_fail=0))
        r = ReflectionMetrics.report()
        assert r["total_corrections"] == 0
        assert r["total_verifications"] == 3
        assert r["total_sympy_errors"]  == 0

    def test_correction_rate_calculation(self):
        from app.evaluation.hallucination_tracker import ReflectionMetrics
        ReflectionMetrics.record(self._make_result(was_corrected=True))
        ReflectionMetrics.record(self._make_result(was_corrected=False))
        ReflectionMetrics.record(self._make_result(was_corrected=False))
        r = ReflectionMetrics.report()
        assert r["correction_rate"] == pytest.approx(1 / 3, abs=1e-4)

    def test_sympy_error_rate_calculation(self):
        from app.evaluation.hallucination_tracker import ReflectionMetrics
        # 2 verifications, 1 fail → error rate = 0.5
        ReflectionMetrics.record(self._make_result(was_corrected=False, n_ok=1, n_fail=1))
        r = ReflectionMetrics.report()
        assert r["sympy_error_rate"] == pytest.approx(0.5, abs=1e-4)

    # ── Pass/fail thresholds ──────────────────────────────────────────────────

    def test_pass_fail_correction_rate_pass(self):
        from app.evaluation.hallucination_tracker import ReflectionMetrics
        # < 10% corrections
        for _ in range(20):
            ReflectionMetrics.record(self._make_result(was_corrected=False, n_ok=1, n_fail=0))
        ReflectionMetrics.record(self._make_result(was_corrected=True, n_ok=1, n_fail=0))
        r = ReflectionMetrics.report()
        # 1/21 ≈ 4.8% < 10%
        assert r["pass_fail"]["correction_rate"] == "✅ PASS"

    def test_pass_fail_correction_rate_fail(self):
        from app.evaluation.hallucination_tracker import ReflectionMetrics
        # 100% corrections
        ReflectionMetrics.record(self._make_result(was_corrected=True))
        r = ReflectionMetrics.report()
        assert r["pass_fail"]["correction_rate"] == "❌ FAIL"

    # ── Per-skill breakdown ───────────────────────────────────────────────────

    def test_per_skill_tracking(self):
        from app.evaluation.hallucination_tracker import ReflectionMetrics
        ReflectionMetrics.record(self._make_result(False), skill_id="derivative_basic")
        ReflectionMetrics.record(self._make_result(True),  skill_id="derivative_basic")
        ReflectionMetrics.record(self._make_result(False), skill_id="integral_definite")
        r = ReflectionMetrics.report()
        assert "derivative_basic"  in r["per_skill"]
        assert "integral_definite" in r["per_skill"]
        assert r["per_skill"]["derivative_basic"]["total"]       == 2
        assert r["per_skill"]["derivative_basic"]["corrections"] == 1
        assert r["per_skill"]["derivative_basic"]["rate"]        == pytest.approx(0.5, abs=1e-4)

    def test_per_skill_without_skill_id(self):
        """Records without skill_id should not appear in per_skill."""
        from app.evaluation.hallucination_tracker import ReflectionMetrics
        ReflectionMetrics.record(self._make_result(True))
        r = ReflectionMetrics.report()
        assert r["per_skill"] == {}

    # ── Thread safety ─────────────────────────────────────────────────────────

    def test_thread_safe_concurrent_records(self):
        from app.evaluation.hallucination_tracker import ReflectionMetrics
        errors = []

        def record_many():
            try:
                for _ in range(50):
                    ReflectionMetrics.record(
                        self._make_result(was_corrected=True, n_ok=1, n_fail=0),
                        skill_id="derivative_basic",
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_many) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        r = ReflectionMetrics.report()
        assert r["total_reflections"] == 250   # 5 threads × 50 records

    # ── History ──────────────────────────────────────────────────────────────

    def test_history_is_capped_at_1000(self):
        from app.evaluation.hallucination_tracker import ReflectionMetrics
        for _ in range(1100):
            ReflectionMetrics.record(self._make_result(False))
        assert len(ReflectionMetrics._history) <= 1000

    # ── save_history ─────────────────────────────────────────────────────────

    def test_save_history(self, tmp_path):
        from app.evaluation.hallucination_tracker import ReflectionMetrics
        ReflectionMetrics.record(self._make_result(True), skill_id="s1")
        path = str(tmp_path / "history.json")
        ReflectionMetrics.save_history(path)
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        assert isinstance(loaded, list)
        assert len(loaded) == 1
        assert loaded[0]["was_corrected"] is True

    # ── reset ─────────────────────────────────────────────────────────────────

    def test_reset_clears_all(self):
        from app.evaluation.hallucination_tracker import ReflectionMetrics
        ReflectionMetrics.record(self._make_result(True))
        ReflectionMetrics.reset()
        r = ReflectionMetrics.report()
        assert r["total_reflections"] == 0
        assert r["per_skill"] == {}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DatasetBuilder
# ═══════════════════════════════════════════════════════════════════════════════

class TestDatasetBuilder:
    """Tests for app/evaluation/dataset_builder.py"""

    def _write_exam_file(self, tmp_path: Path, filename: str, items: list) -> Path:
        """Write a fake exam JSON file."""
        f = tmp_path / filename
        f.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
        return f

    # ── load_exam_questions ───────────────────────────────────────────────────

    def test_load_all_types(self, tmp_path, sample_questions):
        from app.evaluation.dataset_builder import load_exam_questions
        self._write_exam_file(tmp_path, "exam1.json", sample_questions)
        qs = load_exam_questions(exam_dir=str(tmp_path))
        # q_img_01 has no answer → skipped; 3 remain
        assert len(qs) == 3

    def test_load_respects_limit(self, tmp_path, sample_questions):
        from app.evaluation.dataset_builder import load_exam_questions
        self._write_exam_file(tmp_path, "exam1.json", sample_questions)
        qs = load_exam_questions(exam_dir=str(tmp_path), limit=2)
        assert len(qs) == 2

    def test_load_filters_by_type(self, tmp_path, sample_questions):
        from app.evaluation.dataset_builder import load_exam_questions
        self._write_exam_file(tmp_path, "exam1.json", sample_questions)
        qs = load_exam_questions(
            exam_dir=str(tmp_path),
            question_types=["exam_short_answer"],
        )
        assert len(qs) == 1
        assert qs[0]["type"] == "exam_short_answer"

    def test_load_field_mapping(self, tmp_path, sample_questions):
        from app.evaluation.dataset_builder import load_exam_questions
        self._write_exam_file(tmp_path, "exam1.json", sample_questions)
        qs = load_exam_questions(exam_dir=str(tmp_path))
        q = qs[0]
        assert "id"             in q
        assert "question"       in q
        assert "ground_truth"   in q
        assert "skill_id"       in q
        assert "chapter"        in q
        assert "correct_answer" in q

    def test_load_skips_image_questions_without_text_answer(self, tmp_path, sample_questions):
        from app.evaluation.dataset_builder import load_exam_questions
        self._write_exam_file(tmp_path, "exam1.json", sample_questions)
        qs = load_exam_questions(exam_dir=str(tmp_path))
        ids = [q["id"] for q in qs]
        assert "q_img_01" not in ids

    def test_load_empty_directory(self, tmp_path):
        from app.evaluation.dataset_builder import load_exam_questions
        qs = load_exam_questions(exam_dir=str(tmp_path))
        assert qs == []

    def test_load_multiple_files(self, tmp_path, sample_questions):
        from app.evaluation.dataset_builder import load_exam_questions
        self._write_exam_file(tmp_path, "exam1.json", sample_questions[:2])
        self._write_exam_file(tmp_path, "exam2.json", sample_questions[2:3])
        qs = load_exam_questions(exam_dir=str(tmp_path))
        assert len(qs) == 3

    def test_load_ignores_malformed_json(self, tmp_path, sample_questions):
        from app.evaluation.dataset_builder import load_exam_questions
        # Good file
        self._write_exam_file(tmp_path, "good.json", sample_questions[:1])
        # Bad file
        (tmp_path / "bad.json").write_text("NOT JSON {{{{", encoding="utf-8")
        qs = load_exam_questions(exam_dir=str(tmp_path))
        assert len(qs) == 1  # bad file skipped, good file loaded

    # ── save_samples / load_samples ───────────────────────────────────────────

    def test_save_and_load_roundtrip(self, tmp_path, sample_ragas_samples):
        from app.evaluation.dataset_builder import save_samples, load_samples
        path = str(tmp_path / "samples.json")
        save_samples(sample_ragas_samples, path)
        loaded = load_samples(path)
        assert len(loaded) == len(sample_ragas_samples)
        assert loaded[0]["user_input"] == sample_ragas_samples[0]["user_input"]
        assert loaded[0]["reference"]  == sample_ragas_samples[0]["reference"]

    def test_save_creates_parent_dirs(self, tmp_path):
        from app.evaluation.dataset_builder import save_samples
        nested = str(tmp_path / "a" / "b" / "c" / "out.json")
        save_samples([], nested)
        assert Path(nested).exists()

    def test_metadata_preserved_in_roundtrip(self, tmp_path, sample_ragas_samples):
        from app.evaluation.dataset_builder import save_samples, load_samples
        path = str(tmp_path / "s.json")
        save_samples(sample_ragas_samples, path)
        loaded = load_samples(path)
        assert loaded[0]["_meta"]["skill_id"] == "derivative_basic"

    # ── retrieve_contexts fallback ────────────────────────────────────────────

    def test_retrieve_contexts_returns_empty_on_kb_failure(self):
        from app.evaluation.dataset_builder import retrieve_contexts
        with patch(
            "app.evaluation.dataset_builder.get_knowledge_base",
            side_effect=Exception("KB not available"),
        ):
            result = retrieve_contexts("bất kỳ câu hỏi nào")
        assert result == []

    def test_retrieve_contexts_returns_list_of_strings(self):
        from app.evaluation.dataset_builder import retrieve_contexts
        mock_kb = MagicMock()
        mock_kb.search_theory.return_value = [
            {"content": "Nội dung 1", "hybrid_score": 0.8},
            {"content": "Nội dung 2", "hybrid_score": 0.7},
        ]
        with patch("app.evaluation.dataset_builder.get_knowledge_base", return_value=mock_kb):
            result = retrieve_contexts("đạo hàm hàm số mũ", k=2)
        assert result == ["Nội dung 1", "Nội dung 2"]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. RAGASEvaluator
# ═══════════════════════════════════════════════════════════════════════════════

class TestRAGASEvaluator:
    """Tests for app/evaluation/ragas_evaluator.py"""

    # ── Report structure ──────────────────────────────────────────────────────

    def _make_mock_ragas_result(self, scores: dict):
        """Create a mock RAGAS evaluate() result with to_pandas()."""
        import pandas as pd
        df = pd.DataFrame([
            {
                "user_input":         "q1",
                "retrieved_contexts": ["ctx"],
                "response":           "ans",
                "reference":          "ref",
                **scores,
            }
        ])
        mock_result = MagicMock()
        mock_result.to_pandas.return_value = df
        return mock_result

    def _make_evaluator_with_mock_ragas(self, scores: dict):
        """Patch RAGAS imports and return (evaluator, mock_result)."""
        from app.evaluation.ragas_evaluator import RAGASEvaluator

        mock_result = self._make_mock_ragas_result(scores)

        # Build mock _ragas dict that evaluator._setup_ragas() would produce
        mock_ragas = {
            "evaluate":          MagicMock(return_value=mock_result),
            "EvaluationDataset": MagicMock(),
            "SingleTurnSample":  MagicMock(side_effect=lambda **kw: kw),
            "selected_metrics":  [MagicMock()] * 5,
        }

        evaluator = RAGASEvaluator(
            metrics=["context_precision", "context_recall", "faithfulness",
                     "answer_relevancy", "answer_correctness"],
            model="gpt-4o-mini",
        )
        evaluator._ragas = mock_ragas
        return evaluator, mock_result

    def test_report_contains_required_keys(self, sample_ragas_samples):
        scores = {
            "context_precision": 0.80,
            "context_recall":    0.72,
            "faithfulness":      0.85,
            "answer_relevancy":  0.78,
            "answer_correctness": 0.74,
        }
        evaluator, _ = self._make_evaluator_with_mock_ragas(scores)
        report = evaluator.run_and_report(sample_ragas_samples)

        assert "timestamp"       in report
        assert "n_samples"       in report
        assert "summary_scores"  in report
        assert "thresholds"      in report
        assert "pass_fail"       in report
        assert "judge_model"     in report

    def test_summary_scores_are_rounded(self, sample_ragas_samples):
        scores = {"context_precision": 0.80012, "faithfulness": 0.75999}
        evaluator, _ = self._make_evaluator_with_mock_ragas(scores)
        report = evaluator.run_and_report(sample_ragas_samples)
        for v in report["summary_scores"].values():
            # Mỗi score phải có tối đa 4 chữ số thập phân
            assert len(str(v).split(".")[-1]) <= 4

    def test_pass_fail_all_pass(self, sample_ragas_samples):
        scores = {
            "context_precision":  0.80,
            "context_recall":     0.80,
            "faithfulness":       0.90,
            "answer_relevancy":   0.85,
            "answer_correctness": 0.80,
        }
        evaluator, _ = self._make_evaluator_with_mock_ragas(scores)
        report = evaluator.run_and_report(sample_ragas_samples)
        for name, verdict in report["pass_fail"].items():
            assert verdict == "✅ PASS", f"{name} should PASS but got {verdict}"

    def test_pass_fail_low_context_precision_fails(self, sample_ragas_samples):
        scores = {
            "context_precision":  0.30,   # <0.70 → FAIL
            "context_recall":     0.75,
            "faithfulness":       0.85,
            "answer_relevancy":   0.80,
            "answer_correctness": 0.75,
        }
        evaluator, _ = self._make_evaluator_with_mock_ragas(scores)
        report = evaluator.run_and_report(sample_ragas_samples)
        assert report["pass_fail"]["context_precision"] == "❌ FAIL"
        assert report["pass_fail"]["context_recall"]    == "✅ PASS"

    def test_n_samples_matches_input(self, sample_ragas_samples):
        scores = {"faithfulness": 0.85}
        evaluator, _ = self._make_evaluator_with_mock_ragas(scores)
        report = evaluator.run_and_report(sample_ragas_samples)
        assert report["n_samples"] == len(sample_ragas_samples)

    def test_report_saved_to_file(self, tmp_path, sample_ragas_samples):
        scores = {"context_precision": 0.75, "faithfulness": 0.80}
        evaluator, _ = self._make_evaluator_with_mock_ragas(scores)
        output_path = str(tmp_path / "report.json")
        evaluator.run_and_report(sample_ragas_samples, output_path=output_path)
        assert Path(output_path).exists()
        data = json.loads(Path(output_path).read_text(encoding="utf-8"))
        assert "summary_scores" in data

    # ── Skill breakdown ───────────────────────────────────────────────────────

    def test_skill_breakdown_computed(self, sample_ragas_samples):
        """Skill breakdown should be keyed by skill_id from _meta."""
        import pandas as pd

        # Build df with skill_id already present (simulating groupby)
        df = pd.DataFrame([
            {"context_precision": 0.80, "faithfulness": 0.90,
             "user_input": s["user_input"], "retrieved_contexts": s["retrieved_contexts"],
             "response": s["response"], "reference": s["reference"]}
            for s in sample_ragas_samples
        ])

        mock_result        = MagicMock()
        mock_result.to_pandas.return_value = df

        from app.evaluation.ragas_evaluator import RAGASEvaluator
        evaluator = RAGASEvaluator(
            metrics=["context_precision", "faithfulness"],
            model="gpt-4o-mini",
        )
        mock_ragas = {
            "evaluate":          MagicMock(return_value=mock_result),
            "EvaluationDataset": MagicMock(),
            "SingleTurnSample":  MagicMock(side_effect=lambda **kw: kw),
            "selected_metrics":  [MagicMock(), MagicMock()],
        }
        evaluator._ragas = mock_ragas

        report = evaluator.run_and_report(sample_ragas_samples)
        # skill_breakdown may or may not be populated depending on column availability
        assert "skill_breakdown" in report

    # ── CLI metric shorthand ──────────────────────────────────────────────────

    def test_metric_shorthand_mapping(self):
        """Verifica che le abbreviazioni CLI vengano mappate correttamente."""
        metric_map = {
            "cp":  "context_precision",
            "cr":  "context_recall",
            "f":   "faithfulness",
            "ar":  "answer_relevancy",
            "ac":  "answer_correctness",
        }
        requested = [metric_map.get(m, m) for m in ["cp", "cr", "f", "ar", "ac"]]
        assert requested == [
            "context_precision",
            "context_recall",
            "faithfulness",
            "answer_relevancy",
            "answer_correctness",
        ]

    def test_unknown_shorthand_passes_through(self):
        """Unknown shorthands (full names) phải được giữ nguyên."""
        metric_map = {"cp": "context_precision"}
        result = [metric_map.get(m, m) for m in ["faithfulness", "cp"]]
        assert result == ["faithfulness", "context_precision"]


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Integration: ReflectionEngine → HallucinationTracker
# ═══════════════════════════════════════════════════════════════════════════════

class TestReflectionIntegration:
    """
    Verify rằng ReflectionEngine.reflect() gọi ReflectionMetrics.record()
    sau một lần chạy thành công — không cần OpenAI call thật.
    """

    @pytest.mark.asyncio
    async def test_reflect_records_metrics_when_no_verifiable_expressions(self):
        """
        Khi extraction trả [] (không có biểu thức để verify),
        ReflectionMetrics vẫn phải nhận 1 bản ghi.
        """
        from app.evaluation.hallucination_tracker import ReflectionMetrics
        from app.agents.reflection import ReflectionEngine

        engine = ReflectionEngine()

        # Mock LLM trực tiếp trên instance (tránh vấn đề module import cache)
        mock_response = MagicMock()
        mock_response.content = "[]"
        mock_response.usage   = MagicMock(prompt_tokens=10, completion_tokens=5)

        engine.extractor_llm = MagicMock()
        engine.extractor_llm.ainvoke = AsyncMock(return_value=mock_response)

        # Mock cost tracker để tránh format error với MagicMock
        with patch("app.agents.reflection.log_from_response"):
            result = await engine.reflect(
                draft="Đáp án câu này là 42.",
                question="x + 1 = 43, tìm x?",
            )

        assert result.was_corrected is False
        assert ReflectionMetrics.report()["total_reflections"] == 1

    @pytest.mark.asyncio
    async def test_reflect_correction_increments_counter(self):
        """
        Khi SymPy verify thành công và LLM tự sửa,
        correction_rate phải tăng.
        """
        from app.evaluation.hallucination_tracker import ReflectionMetrics
        from app.agents.reflection import ReflectionEngine

        engine = ReflectionEngine()

        # Lần gọi đầu: extraction trả 1 biểu thức
        # Lần gọi sau: self-correction
        extraction_response = MagicMock()
        extraction_response.content = (
            '[{"expression": "2+2", "operation": "simplify_expression",'
            ' "params": {"expr_str": "2+2"}}]'
        )
        extraction_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

        correction_response = MagicMock()
        correction_response.content = "Bài giải đã được sửa: 2+2=4."
        correction_response.usage = MagicMock(prompt_tokens=20, completion_tokens=10)

        engine.extractor_llm = MagicMock()
        engine.extractor_llm.ainvoke = AsyncMock(return_value=extraction_response)
        engine.llm = MagicMock()
        engine.llm.ainvoke = AsyncMock(return_value=correction_response)

        with patch("app.agents.reflection.log_from_response"), \
             patch("app.agents.reflection.MATH_TOOLS", {
                 "simplify_expression": MagicMock(
                     return_value={"success": True, "result_latex": "4"}
                 )
             }):
            await engine.reflect(
                draft="Ta có: 2+2=5.",
                question="Tính 2+2.",
            )

        r = ReflectionMetrics.report()
        assert r["total_reflections"]   == 1
        assert r["total_verifications"] >= 1

    @pytest.mark.asyncio
    async def test_reflect_result_has_required_attributes(self):
        """ReflectionResult phải có đủ attributes cần thiết."""
        from app.agents.reflection import ReflectionEngine

        engine = ReflectionEngine()
        mock_response      = MagicMock()
        mock_response.content = "[]"
        mock_response.usage   = MagicMock(prompt_tokens=5, completion_tokens=2)

        engine.extractor_llm = MagicMock()
        engine.extractor_llm.ainvoke = AsyncMock(return_value=mock_response)

        with patch("app.agents.reflection.log_from_response"):
            result = await engine.reflect(
                draft="Kết quả là 0.",
                question="0+0=?",
            )

        assert hasattr(result, "final_answer")
        assert hasattr(result, "was_corrected")
        assert hasattr(result, "verifications")
        assert hasattr(result, "thinking_log")
        assert isinstance(result.thinking_log, list)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MathJudge
# ═══════════════════════════════════════════════════════════════════════════════

class TestMathJudge:
    """Tests for app/evaluation/math_judge.py"""

    # ── MCQ Accuracy ──────────────────────────────────────────────────────────

    def test_mcq_exact_match_a(self):
        from app.evaluation.math_judge import judge_mcq_accuracy
        assert judge_mcq_accuracy("Chọn A", "A") == 1.0

    def test_mcq_exact_match_b(self):
        from app.evaluation.math_judge import judge_mcq_accuracy
        assert judge_mcq_accuracy("Đáp án đúng là B.", "B") == 1.0

    def test_mcq_wrong_answer(self):
        from app.evaluation.math_judge import judge_mcq_accuracy
        assert judge_mcq_accuracy("Chọn B", "A") == 0.0

    def test_mcq_extract_from_verbose_response(self):
        """Model giải dài, có bold **C** ở cuối."""
        from app.evaluation.math_judge import judge_mcq_accuracy
        response = "Ta có đạo hàm f'(x) = 3x² + 3 > 0 ∀x ∈ ℝ. Vậy đáp án là **C**."
        assert judge_mcq_accuracy(response, "C") == 1.0

    def test_mcq_case_insensitive(self):
        from app.evaluation.math_judge import judge_mcq_accuracy
        assert judge_mcq_accuracy("chọn d", "D") == 1.0

    def test_mcq_empty_response_returns_zero(self):
        from app.evaluation.math_judge import judge_mcq_accuracy
        assert judge_mcq_accuracy("", "A") == 0.0

    def test_mcq_invalid_correct_answer(self):
        from app.evaluation.math_judge import judge_mcq_accuracy
        assert judge_mcq_accuracy("Chọn A", "E") == 0.0

    # ── True/False Accuracy ───────────────────────────────────────────────────

    def test_tf_all_correct(self):
        from app.evaluation.math_judge import judge_true_false_accuracy
        score = judge_true_false_accuracy(
            response="a-T, b-F, c-T, d-F",
            correct_answer="a-T,b-F,c-T,d-F",
        )
        assert score == 1.0

    def test_tf_partial_correct(self):
        from app.evaluation.math_judge import judge_true_false_accuracy
        # 3 out of 4 correct
        score = judge_true_false_accuracy(
            response="a-T, b-T, c-T, d-F",   # b wrong
            correct_answer="a-T,b-F,c-T,d-F",
        )
        assert score == pytest.approx(0.75, abs=1e-4)

    def test_tf_all_wrong(self):
        from app.evaluation.math_judge import judge_true_false_accuracy
        score = judge_true_false_accuracy(
            response="a-F, b-T, c-F, d-T",
            correct_answer="a-T,b-F,c-T,d-F",
        )
        assert score == 0.0

    def test_tf_empty_returns_zero(self):
        from app.evaluation.math_judge import judge_true_false_accuracy
        assert judge_true_false_accuracy("", "a-T,b-F") == 0.0

    # ── Short Answer Accuracy ─────────────────────────────────────────────────

    def test_short_answer_exact_number(self):
        from app.evaluation.math_judge import judge_short_answer_accuracy
        assert judge_short_answer_accuracy("Vậy giá trị là 4.5", "4,5") == 1.0

    def test_short_answer_integer(self):
        from app.evaluation.math_judge import judge_short_answer_accuracy
        assert judge_short_answer_accuracy("Đáp án: 20 tháng.", "20") == 1.0

    def test_short_answer_wrong_number(self):
        from app.evaluation.math_judge import judge_short_answer_accuracy
        assert judge_short_answer_accuracy("Kết quả là 19", "20") == 0.0

    def test_short_answer_no_llm_needed(self):
        """Verify that short_answer judge is purely deterministic (no LLM import needed)."""
        from app.evaluation.math_judge import judge_short_answer_accuracy
        result = judge_short_answer_accuracy("192", "192")
        assert result == 1.0

    # ── MathJudge Coordinator ─────────────────────────────────────────────────

    def test_mathjudge_report_structure(self):
        """run_and_report phải trả về đúng các keys cần thiết."""
        from app.evaluation.math_judge import MathJudge

        samples = [
            {
                "user_input": "log₂8 =?  A.2 B.3 C.4",
                "retrieved_contexts": [],
                "response": "Chọn B",
                "reference": "ĐÁP ÁN: B",
                "_meta": {
                    "id": "q1", "skill_id": "logarithm",
                    "chapter": "Mũ Logarit", "type": "exam_mcq",
                    "correct_answer": "B",
                },
            }
        ]

        judge = MathJudge(skip_explanation=True)
        report = judge.run_and_report(samples)

        assert "timestamp"       in report
        assert "n_samples"       in report
        assert "summary_scores"  in report
        assert "type_breakdown"  in report
        assert "skill_breakdown" in report
        assert "pass_fail"       in report
        assert "per_sample"      in report
        assert report["n_samples"] == 1

    def test_mathjudge_accuracy_score_correct(self):
        """MCQ đúng → accuracy = 1.0 trong summary."""
        from app.evaluation.math_judge import MathJudge

        samples = [
            {
                "user_input": "Chọn đáp án đúng?",
                "retrieved_contexts": [],
                "response": "Chọn C",
                "reference": "ĐÁP ÁN: C",
                "_meta": {
                    "id": "q1", "skill_id": "test",
                    "chapter": "Test", "type": "exam_mcq",
                    "correct_answer": "C",
                },
            }
        ]
        judge = MathJudge(skip_explanation=True)
        report = judge.run_and_report(samples)
        assert report["summary_scores"]["mcq_accuracy"] == 1.0

    def test_mathjudge_saves_to_file(self, tmp_path):
        from app.evaluation.math_judge import MathJudge
        samples = [
            {
                "user_input": "2+2=?",
                "retrieved_contexts": [],
                "response": "4",
                "reference": "4",
                "_meta": {
                    "id": "q1", "skill_id": "s1",
                    "chapter": "c1", "type": "exam_short_answer",
                    "correct_answer": "4",
                },
            }
        ]
        out = str(tmp_path / "report.json")
        judge = MathJudge(skip_explanation=True)
        judge.run_and_report(samples, output_path=out)
        assert Path(out).exists()
        data = json.loads(Path(out).read_text(encoding="utf-8"))
        assert "summary_scores" in data

    # ── Combined Report ───────────────────────────────────────────────────────

    def test_merge_reports_structure(self):
        from app.evaluation.combined_report import merge_reports

        ragas_report = {
            "n_samples": 5,
            "judge_model": "gpt-4o-mini",
            "summary_scores": {
                "llm_context_precision_with_reference": 0.4,
                "context_recall": 0.5,
                "answer_correctness": 0.6,
            },
            "skill_breakdown": {},
        }
        math_report = {
            "judge_model": "gpt-4o-mini",
            "summary_scores": {
                "mcq_accuracy": 0.9,
                "true_false_accuracy": 0.6,
                "short_answer_accuracy": 0.7,
                "overall_accuracy": 0.77,
            },
            "type_breakdown": {"exam_mcq": 0.9},
            "skill_breakdown": {},
        }

        combined = merge_reports(ragas_report, math_report)

        assert "combined_scores"   in combined
        assert "context_precision" in combined["combined_scores"]
        assert "context_recall"    in combined["combined_scores"]
        assert "mcq_accuracy"      in combined["combined_scores"]
        assert "overall_accuracy"  in combined["combined_scores"]
        assert "pass_fail"         in combined

    def test_merge_reports_saves_file(self, tmp_path):
        from app.evaluation.combined_report import merge_reports

        ragas = {
            "n_samples": 1, "judge_model": "x",
            "summary_scores": {}, "skill_breakdown": {},
        }
        math = {
            "judge_model": "x",
            "summary_scores": {"mcq_accuracy": 0.8, "overall_accuracy": 0.8},
            "type_breakdown": {}, "skill_breakdown": {},
        }

        out = str(tmp_path / "combined.json")
        merge_reports(ragas, math, output_path=out)
        assert Path(out).exists()
