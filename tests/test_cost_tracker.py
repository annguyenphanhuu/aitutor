"""
Tests for Cost Tracker utility.

Module: app/utils/cost_tracker.py
Covers:
- _get_price() — pricing lookup with fallback
- log_call() — token usage logging
- log_from_response() — LangChain response parsing
- session_summary() — summary formatting
- Session accumulator reset behaviour
"""

from unittest.mock import MagicMock
from app.utils.cost_tracker import (
    _get_price,
    log_call,
    log_from_response,
    session_summary,
    _session,
    _PRICING,
)


# ━━ _get_price() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGetPrice:

    def test_known_model(self):
        price = _get_price("gpt-4o")
        assert "input" in price
        assert "output" in price
        assert price["input"] > 0
        assert price["output"] > 0

    def test_gpt4o_mini(self):
        price = _get_price("gpt-4o-mini")
        assert price["input"] < _get_price("gpt-4o")["input"]

    def test_unknown_model_fallback(self):
        """Unknown model should fallback to gpt-4o pricing."""
        price = _get_price("some-future-model-xyz")
        fallback = _PRICING["gpt-4o"]
        assert price == fallback

    def test_all_models_in_pricing(self):
        for model in _PRICING:
            price = _get_price(model)
            assert "input" in price
            assert "output" in price


# ━━ log_call() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestLogCall:

    def setup_method(self):
        # Reset session counters
        _session.total_input = 0
        _session.total_output = 0
        _session.total_cost_usd = 0.0
        _session.call_count = 0

    def test_returns_cost(self):
        cost = log_call(agent="Test", model="gpt-4o", input_tokens=100, output_tokens=50)
        assert isinstance(cost, float)
        assert cost > 0

    def test_updates_session_counters(self):
        log_call(agent="Test", model="gpt-4o", input_tokens=100, output_tokens=50)
        assert _session.total_input == 100
        assert _session.total_output == 50
        assert _session.call_count == 1
        assert _session.total_cost_usd > 0

    def test_accumulates_multiple_calls(self):
        log_call(agent="A", model="gpt-4o", input_tokens=100, output_tokens=50)
        log_call(agent="B", model="gpt-4o", input_tokens=200, output_tokens=100)
        assert _session.total_input == 300
        assert _session.total_output == 150
        assert _session.call_count == 2

    def test_cost_calculation_formula(self):
        """Cost = (input * input_price + output * output_price) / 1000."""
        price = _get_price("gpt-4o")
        cost = log_call(agent="Test", model="gpt-4o", input_tokens=1000, output_tokens=1000)
        expected = (1000 * price["input"] + 1000 * price["output"]) / 1000
        assert abs(cost - expected) < 1e-10

    def test_with_extra_parameter(self):
        cost = log_call(agent="Test", model="gpt-4o", input_tokens=100,
                        output_tokens=50, extra="intent=explain")
        assert cost > 0

    def test_zero_tokens(self):
        cost = log_call(agent="Test", model="gpt-4o", input_tokens=0, output_tokens=0)
        assert cost == 0.0


# ━━ log_from_response() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestLogFromResponse:

    def setup_method(self):
        _session.total_input = 0
        _session.total_output = 0
        _session.total_cost_usd = 0.0
        _session.call_count = 0

    def test_langchain_usage_metadata(self):
        """Test with LangChain-style usage_metadata dict."""
        response = MagicMock()
        response.usage_metadata = {"input_tokens": 500, "output_tokens": 200}
        cost = log_from_response(agent="Teacher", model="gpt-4o", response=response)
        assert cost > 0
        assert _session.total_input == 500
        assert _session.total_output == 200

    def test_legacy_response_metadata(self):
        """Test with older LangChain response_metadata format."""
        response = MagicMock()
        response.usage_metadata = None
        response.response_metadata = {
            "token_usage": {"prompt_tokens": 300, "completion_tokens": 100}
        }
        cost = log_from_response(agent="Assessor", model="gpt-4o", response=response)
        assert cost > 0
        assert _session.total_input == 300

    def test_no_usage_info(self):
        """Should gracefully handle response with no usage info."""
        response = MagicMock()
        response.usage_metadata = None
        response.response_metadata = {}
        cost = log_from_response(agent="Test", model="gpt-4o", response=response)
        # Should not crash; tokens default to 0
        assert cost == 0.0


# ━━ session_summary() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestSessionSummary:

    def setup_method(self):
        _session.total_input = 0
        _session.total_output = 0
        _session.total_cost_usd = 0.0
        _session.call_count = 0

    def test_empty_session(self):
        summary = session_summary()
        assert "0 calls" in summary
        assert "$0.0000" in summary

    def test_after_calls(self):
        log_call(agent="A", model="gpt-4o", input_tokens=500, output_tokens=200)
        log_call(agent="B", model="gpt-4o", input_tokens=300, output_tokens=100)
        summary = session_summary()
        assert "2 calls" in summary
        assert "in=800" in summary
        assert "out=300" in summary

    def test_format_contains_emoji(self):
        summary = session_summary()
        assert "📊" in summary
