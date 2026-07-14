"""
Tests for GraphRAG module.

Module: app/rag/graph_rag.py
Covers:
- GraphRAGResult — container class
- GraphRAGResult.all_docs — combined document list
- GraphRAGResult.build_context_text() — prompt context builder
- GraphRAGResult.build_gap_warning() — gap warning builder
"""

from langchain.schema import Document
from app.rag.graph_rag import GraphRAGResult


# ━━ GraphRAGResult ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGraphRAGResult:

    def _make_doc(self, content: str, **meta) -> Document:
        return Document(page_content=content, metadata=meta)

    def test_basic_construction(self):
        r = GraphRAGResult(
            main_docs=[self._make_doc("Main content")],
            prereq_docs=[],
            weak_skills=[],
        )
        assert len(r.main_docs) == 1
        assert len(r.prereq_docs) == 0
        assert len(r.weak_skills) == 0

    def test_all_docs_combines(self):
        r = GraphRAGResult(
            main_docs=[self._make_doc("A"), self._make_doc("B")],
            prereq_docs=[self._make_doc("C")],
            weak_skills=[],
        )
        all_docs = r.all_docs
        assert len(all_docs) == 3
        assert all_docs[0].page_content == "A"
        assert all_docs[2].page_content == "C"

    def test_all_docs_empty(self):
        r = GraphRAGResult(main_docs=[], prereq_docs=[], weak_skills=[])
        assert r.all_docs == []


# ━━ build_context_text() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestBuildContextText:

    def _make_doc(self, content: str, **meta) -> Document:
        return Document(page_content=content, metadata=meta)

    def test_main_docs_only(self):
        r = GraphRAGResult(
            main_docs=[self._make_doc("Lý thuyết đạo hàm...")],
            prereq_docs=[],
            weak_skills=[],
        )
        text = r.build_context_text()
        assert "TÀI LIỆU CHÍNH" in text
        assert "Lý thuyết đạo hàm..." in text

    def test_with_prereqs(self):
        r = GraphRAGResult(
            main_docs=[self._make_doc("Tích phân...")],
            prereq_docs=[self._make_doc("Đạo hàm cơ bản...")],
            weak_skills=[{"skill_id": "derivative_basic", "skill_name": "Đạo hàm cơ bản", "mastery": 0.2}],
        )
        text = r.build_context_text()
        assert "TÀI LIỆU CHÍNH" in text
        assert "KIẾN THỨC NỀN TẢNG" in text
        assert "Đạo hàm cơ bản" in text
        assert "20%" in text

    def test_empty_returns_default(self):
        r = GraphRAGResult(main_docs=[], prereq_docs=[], weak_skills=[])
        text = r.build_context_text()
        assert "Không tìm thấy" in text

    def test_multiple_main_docs(self):
        r = GraphRAGResult(
            main_docs=[self._make_doc("Doc A"), self._make_doc("Doc B")],
            prereq_docs=[],
            weak_skills=[],
        )
        text = r.build_context_text()
        assert "Doc A" in text
        assert "Doc B" in text


# ━━ build_gap_warning() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestBuildGapWarning:

    def test_no_weak_skills(self):
        r = GraphRAGResult(main_docs=[], prereq_docs=[], weak_skills=[])
        assert r.build_gap_warning() == ""

    def test_single_weak_skill(self):
        r = GraphRAGResult(
            main_docs=[], prereq_docs=[],
            weak_skills=[{"skill_id": "derivative_basic", "skill_name": "Đạo hàm cơ bản", "mastery": 0.2}],
        )
        warning = r.build_gap_warning()
        assert "⚠️" in warning
        assert "Đạo hàm cơ bản" in warning
        assert "20%" in warning
        assert "≥50%" in warning

    def test_multiple_weak_skills(self):
        r = GraphRAGResult(
            main_docs=[], prereq_docs=[],
            weak_skills=[
                {"skill_id": "a", "skill_name": "Skill A", "mastery": 0.1},
                {"skill_id": "b", "skill_name": "Skill B", "mastery": 0.3},
            ],
        )
        warning = r.build_gap_warning()
        assert "Skill A" in warning
        assert "Skill B" in warning
        assert warning.count("•") == 2
