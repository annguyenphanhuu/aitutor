"""Regression tests for taxonomy loading and real-world query rewriting."""

from app.knowledge_tracing.skill_graph import SKILLS
from app.rag.graph_rag import build_search_queries, merge_search_results
from app.rag.query_expander import QueryExpander


def test_taxonomy_is_derived_from_skill_graph_and_content_registry():
    expander = QueryExpander(api_key="test")

    assert set(SKILLS).issubset(set(expander.known_skill_ids))
    assert "geometry_vectors" in expander.known_skill_ids
    assert "geometry_vector" not in expander.known_skill_ids
    assert "Hàm số" in expander.known_chapters
    assert "Lượng giác" in expander.known_chapters


def test_validate_normalizes_taxonomy_and_sanitizes_rewrite():
    expander = QueryExpander(api_key="test")
    result = expander._validate({
        "chapters": ["Nguyên hàm – Tích phân"],
        "skill_ids": ["geometry_vectors", "not_a_real_skill"],
        "formula_ids": [],
        "rag_query": "  xác suất có điều kiện\n  công thức Bayes  ",
    })

    assert result.chapters == ["Nguyên hàm - Tích phân"]
    assert result.skill_ids == ["geometry_vectors"]
    assert result.rag_query == "xác suất có điều kiện công thức Bayes"


def test_validate_rejects_non_string_rewrite():
    expander = QueryExpander(api_key="test")
    result = expander._validate({"rag_query": ["not", "a", "query"]})
    assert result.rag_query == ""


def test_build_search_queries_prefers_rewrite_but_keeps_original_fallback():
    queries = build_search_queries(
        "Một kho có sản phẩm lỗi",
        "xác suất có điều kiện Bayes",
    )
    assert queries == ["xác suất có điều kiện Bayes", "Một kho có sản phẩm lỗi"]
    assert build_search_queries("  Cùng query ", "cùng   query") == ["cùng query"]


def test_merge_search_results_rewards_docs_found_by_both_queries():
    rewritten = [
        {"id": "shared", "content": "Bayes", "hybrid_score": 0.5},
        {"id": "rewrite-only", "content": "Xác suất", "hybrid_score": 0.4},
    ]
    original = [
        {"id": "shared", "content": "Bayes", "hybrid_score": 0.45},
        {"id": "raw-only", "content": "Kho hàng", "hybrid_score": 0.3},
    ]

    merged = merge_search_results([rewritten, original])
    by_id = {item["id"]: item for item in merged}

    assert set(by_id) == {"shared", "rewrite-only", "raw-only"}
    assert by_id["shared"]["hybrid_score"] > 0.5
    assert by_id["shared"]["_matched_query_indexes"] == [0, 1]

