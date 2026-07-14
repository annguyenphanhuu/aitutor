"""Tests for the shared Teacher prompt preparation path."""

from unittest.mock import MagicMock

from langchain.schema import AIMessage, HumanMessage

from app.agents.teacher_agent import TeacherAgent
from app.rag.graph_rag import GraphRAGResult


def _make_agent() -> TeacherAgent:
    agent = TeacherAgent.__new__(TeacherAgent)
    agent.graph_retriever = MagicMock()
    agent._prompt_socratic = (
        "CTX={context}|SKILLS={skill_names_list}|FORMULAS={formulas_list}|"
        "LEVEL={mastery_level}|GAPS={prerequisite_gaps}|FEW={few_shot_block}"
    )
    agent._prompt_exam = "EXAM {context} {skill_names_list} {formulas_list} {mastery_level} {few_shot_block}"
    agent._prompt_answer = "ANSWER {context} {skill_names_list} {formulas_list} {mastery_level} {few_shot_block}"
    return agent


def test_prepare_and_render_prompt_uses_one_shared_context_path():
    agent = _make_agent()
    agent.graph_retriever.retrieve.return_value = GraphRAGResult([], [], [])

    prepared = agent._prepare_prompt_context(
        "đạo hàm",
        skill_id="derivative_basic",
        skill_ids=["derivative_basic"],
        formula_ids=[],
        masteries={"derivative_basic": 0.4},
        p_mastery=0.4,
        prerequisite_gaps=[{"skill_name": "Hàm số"}],
    )
    prompt = agent._build_system_prompt(
        prepared,
        mode="socratic",
        mastery_level="beginner",
        question_type="exam_mcq",
    )

    agent.graph_retriever.retrieve.assert_called_once_with(
        query="đạo hàm",
        skill_id="derivative_basic",
        masteries={"derivative_basic": 0.4},
    )
    assert "Không tìm thấy tài liệu liên quan." in prompt
    assert "Hàm số" in prompt
    assert "<answer>X</answer>" in prompt


def test_history_messages_are_bounded_and_typed():
    history = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": str(index)}
        for index in range(12)
    ]
    messages = TeacherAgent._history_messages(history)

    assert len(messages) == 10
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert messages[0].content == "2"
