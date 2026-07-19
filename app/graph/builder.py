"""Lắp ráp tutor graph từ TutorNodes.

QUAN TRỌNG: mọi node nối tuần tự — TUYỆT ĐỐI không thêm nhánh chạy song song.
Các node dùng chung một SQLAlchemy AsyncSession (qua config), và AsyncSession
raise khi bị dùng đồng thời.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.graph.nodes import TutorNodes, route_gradable, route_intent
from app.graph.state import TutorState


def build_tutor_graph(nodes: TutorNodes):
    """Build và compile tutor graph. Gọi 1 lần lúc khởi tạo engine."""
    graph = StateGraph(TutorState)

    graph.add_node("classify", nodes.classify)
    graph.add_node("hydrate", nodes.hydrate)
    graph.add_node("off_topic", nodes.off_topic)
    graph.add_node("plan", nodes.plan)
    graph.add_node("quiz", nodes.quiz)
    graph.add_node("review", nodes.review)
    graph.add_node("diagnostic", nodes.diagnostic)
    graph.add_node("visualize", nodes.visualize)
    graph.add_node("teach", nodes.teach)
    graph.add_node("grade_extract", nodes.grade_extract)
    graph.add_node("grade_verify", nodes.grade_verify)
    graph.add_node("grade_feedback", nodes.grade_feedback)
    graph.add_node("grade_policy", nodes.grade_policy)
    graph.add_node("finalize", nodes.finalize)

    graph.add_edge(START, "classify")
    graph.add_edge("classify", "hydrate")

    graph.add_conditional_edges(
        "hydrate",
        route_intent,
        {
            "off_topic": "off_topic",
            "plan": "plan",
            "quiz": "quiz",
            "review": "review",
            "diagnostic": "diagnostic",
            "visualize": "visualize",
            "teach": "teach",
            "grade_extract": "grade_extract",
        },
    )

    # Abstain handoff: không có bài làm của học sinh → giải thích thay vì chấm
    graph.add_conditional_edges(
        "grade_extract",
        route_gradable,
        {"grade_verify": "grade_verify", "teach": "teach"},
    )
    graph.add_edge("grade_verify", "grade_feedback")
    graph.add_edge("grade_feedback", "grade_policy")
    graph.add_edge("grade_policy", "finalize")

    for terminal in ("off_topic", "plan", "quiz", "review", "diagnostic", "visualize", "teach"):
        graph.add_edge(terminal, "finalize")

    graph.add_edge("finalize", END)

    return graph.compile()
