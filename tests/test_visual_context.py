"""Tests for choosing focused graph inputs for the VLM."""

from types import SimpleNamespace

from app.exam_solver.visual_context import select_question_visuals
from app.agents.teacher_agent import needs_visual_feature_pass


def _image(raw: bytes, page_num: int, label: str, width: int = 100, height: int = 100):
    return SimpleNamespace(
        image_bytes=raw,
        page_num=page_num,
        label=label,
        width=width,
        height=height,
        mime_type="image/png",
    )


def test_prefers_relevant_cropped_graphs_over_whole_page():
    visuals = select_question_visuals(
        figure_pages=[2],
        page_images={2: b"whole-page"},
        extracted_images=[
            _image(b"wrong-page", page_num=0, label="graph"),
            _image(b"table", page_num=1, label="table"),
            _image(b"graph", page_num=1, label="graph"),
        ],
    )

    assert [visual.image_bytes for visual in visuals] == [b"graph", b"table"]
    assert all(visual.page_num == 2 for visual in visuals)


def test_falls_back_to_relevant_page_when_no_crop_exists():
    visuals = select_question_visuals(
        figure_pages=[3],
        page_images={2: b"page-2", 3: b"page-3"},
        extracted_images=[],
    )
    assert len(visuals) == 1
    assert visuals[0].image_bytes == b"page-3"
    assert visuals[0].label == "page"


def test_deduplicates_identical_extracted_images():
    visuals = select_question_visuals(
        figure_pages=[1],
        page_images={1: b"whole"},
        extracted_images=[
            _image(b"same", page_num=0, label="graph"),
            _image(b"same", page_num=0, label="graph"),
        ],
    )
    assert len(visuals) == 1


def test_ignores_invalid_page_metadata_and_uses_available_page():
    visuals = select_question_visuals(
        figure_pages=[None, "unknown", 2, "2"],
        page_images={2: b"page-2"},
        extracted_images=[_image(b"bad-page", page_num="unknown", label="graph")],
    )

    assert [visual.image_bytes for visual in visuals] == [b"page-2"]


def test_graphs_trigger_dedicated_visual_evidence_pass():
    assert needs_visual_feature_pass("Cho đồ thị hàm số như hình bên") is True
    assert needs_visual_feature_pass("", image_labels=["graph"]) is True
    assert needs_visual_feature_pass("Đọc đề bài", image_labels=["photo"]) is False
