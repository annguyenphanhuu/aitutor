"""Tests SSE frame contract của GraphTutorEngine.handle_message_stream.

Contract (giống Orchestrator.handle_message_stream):
  meta → token* → done          (teach: nhiều token, done gọn)
  meta → token → done giàu meta (intent không stream: 1 token full text,
                                 done có skill_id/visualization/...)
  error frame khi exception.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.contracts import GradingExtraction, IntentClassification


@pytest.fixture
def engine():
    with patch("app.graph.nodes.TeacherAgent") as MockTeacher, \
         patch("app.graph.nodes.PlannerAgent"), \
         patch("app.graph.nodes.VisualizerAgent") as MockVisualizer, \
         patch("app.graph.nodes.GradingExtractor") as MockExtractor, \
         patch("app.graph.nodes.AsyncOpenAI"), \
         patch("app.graph.nodes.ChatOpenAI") as MockChat:

        async def fake_stream(**kwargs):
            for token in ["Xin ", "chào ", "em!"]:
                yield token

        MockTeacher.return_value.respond_stream = MagicMock(side_effect=fake_stream)
        MockTeacher.return_value.respond = AsyncMock(return_value="blocking response")
        MockVisualizer.return_value.generate_function_plot = MagicMock(
            return_value={"vis_type": "plot", "data": {"traces": []}}
        )
        MockExtractor.return_value.extract = AsyncMock(
            return_value=GradingExtraction(gradable=False)
        )
        MockChat.return_value.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value={"raw": None, "parsed": IntentClassification(), "parsing_error": None}
        )

        from app.graph.service import GraphTutorEngine
        yield GraphTutorEngine()


def _stub_classifier(engine, classification):
    engine.nodes.classifier_llm = MagicMock()
    engine.nodes.classifier_llm.ainvoke = AsyncMock(
        return_value={"raw": None, "parsed": classification, "parsing_error": None}
    )


async def _collect_frames(engine, db, message, **kwargs):
    frames = []
    async for frame_json in engine.handle_message_stream(
        db=db, message=message, **kwargs
    ):
        frames.append(json.loads(frame_json))
    return frames


class TestStreamedTeachIntent:
    async def test_frame_order_meta_tokens_done(self, engine, db_session):
        _stub_classifier(engine, IntentClassification(
            intent="explain", skill_id="derivative_basic",
            skill_ids=["derivative_basic"],
        ))
        frames = await _collect_frames(engine, db_session, "đạo hàm là gì?")

        types = [f["type"] for f in frames]
        assert types[0] == "meta"
        assert types[-1] == "done"
        assert all(t == "token" for t in types[1:-1])
        assert len([t for t in types if t == "token"]) == 3

    async def test_meta_frame_fields(self, engine, db_session):
        _stub_classifier(engine, IntentClassification(
            intent="explain", skill_id="derivative_basic",
            skill_ids=["derivative_basic"], formula_ids=["f1"],
        ))
        frames = await _collect_frames(engine, db_session, "đạo hàm là gì?")
        meta = frames[0]
        assert set(meta.keys()) == {
            "type", "skill_id", "skill_ids", "formula_ids",
            "skill_name", "mastery_level", "mode_used", "intent",
        }
        assert meta["skill_id"] == "derivative_basic"
        assert meta["intent"] == "explain"

    async def test_done_frame_is_plain_for_streamed_teach(self, engine, db_session):
        # Giữ bất đối xứng của engine cũ: teach stream → done frame gọn
        _stub_classifier(engine, IntentClassification(intent="explain"))
        frames = await _collect_frames(engine, db_session, "giúp em")
        done = frames[-1]
        assert set(done.keys()) == {"type", "full_response"}
        assert done["full_response"] == "Xin chào em!"


class TestNonStreamableIntent:
    async def test_single_token_and_rich_done(self, engine, db_session):
        _stub_classifier(engine, IntentClassification(intent="review"))
        frames = await _collect_frames(engine, db_session, "ôn tập")

        types = [f["type"] for f in frames]
        assert types == ["meta", "token", "done"]
        done = frames[-1]
        # done frame giàu metadata — authoritative cho non-streamable intents
        assert set(done.keys()) == {
            "type", "full_response", "skill_id", "skill_ids", "formula_ids",
            "skill_name", "mastery_level", "mode_used", "visualization",
        }
        assert done["mode_used"] == "review"
        assert done["full_response"] == frames[1]["content"]

    async def test_visualize_done_carries_visualization(self, engine, db_session):
        _stub_classifier(engine, IntentClassification(intent="visualize"))
        engine.nodes._extract_plottable_expr_llm = AsyncMock(return_value="x**2")
        frames = await _collect_frames(engine, db_session, "vẽ đồ thị y = x^2")
        done = frames[-1]
        assert done["mode_used"] == "visualize"
        assert done["visualization"] == {"vis_type": "plot", "data": {"traces": []}}


class TestStreamErrors:
    async def test_error_frame_on_exception(self, engine, db_session):
        _stub_classifier(engine, IntentClassification(intent="explain"))

        async def broken_stream(**kwargs):
            yield "một token"
            raise RuntimeError("LLM died")

        engine.nodes.teacher.respond_stream = MagicMock(side_effect=broken_stream)
        frames = await _collect_frames(engine, db_session, "giúp em")
        assert frames[-1]["type"] == "error"
        assert "LLM died" in frames[-1]["message"]
