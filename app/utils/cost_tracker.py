"""Cost tracker — log token usage and estimated USD cost to console."""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("cost")

# ── Pricing table (USD per 1 000 tokens) ────────────────────────────────────
# Update these when OpenAI changes prices.
# Source: https://openai.com/api/pricing/
_PRICING: dict[str, dict[str, float]] = {
    # model-name : {input, output}  (per 1k tokens)
    "gpt-4o":                     {"input": 0.0025,  "output": 0.01},
    "gpt-4o-mini":                {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo":                {"input": 0.01,    "output": 0.03},
    "gpt-4":                      {"input": 0.03,    "output": 0.06},
    "gpt-3.5-turbo":              {"input": 0.0005,  "output": 0.0015},
    "o1":                         {"input": 0.015,   "output": 0.06},
    "o1-mini":                    {"input": 0.003,   "output": 0.012},
    "o3-mini":                    {"input": 0.0011,  "output": 0.0044},
    # gpt-5.x — placeholder until official pricing is released
    "gpt-5.1-2025-11-13":         {"input": 0.01,    "output": 0.03},
    "gpt-5.4-2026-03-05":         {"input": 0.01,    "output": 0.03},
}

# Session accumulator (reset on server restart)
@dataclass
class _Session:
    total_input: int = 0
    total_output: int = 0
    total_cost_usd: float = 0.0
    call_count: int = 0

_session = _Session()


def _get_price(model: str) -> dict[str, float]:
    """Return per-1k-token price for a model, falling back to gpt-4o."""
    # Try exact match first, then prefix match
    if model in _PRICING:
        return _PRICING[model]
    for key in _PRICING:
        if model.startswith(key) or key.startswith(model.split("-")[0]):
            return _PRICING[key]
    logger.warning("💸 Unknown model '%s', using gpt-4o pricing as fallback.", model)
    return _PRICING["gpt-4o"]


def log_call(
    *,
    agent: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    extra: Optional[str] = None,
) -> float:
    """
    Print token usage and cost to console. Returns cost in USD for this call.

    Args:
        agent:         Human-readable label (e.g. "Classifier", "Teacher")
        model:         OpenAI model name
        input_tokens:  Prompt tokens used
        output_tokens: Completion tokens used
        extra:         Optional extra info to append (e.g. intent)
    """
    price = _get_price(model)
    cost = (input_tokens * price["input"] + output_tokens * price["output"]) / 1000.0

    # Update session totals
    _session.total_input  += input_tokens
    _session.total_output += output_tokens
    _session.total_cost_usd += cost
    _session.call_count += 1

    extra_str = f" | {extra}" if extra else ""
    log_msg = (
        f"💸 [{agent}] {model}  "
        f"in={input_tokens} out={output_tokens} → ${cost:.5f}{extra_str}  "
        f"(session: {_session.call_count} calls, ${_session.total_cost_usd:.4f})"
    )
    print(log_msg)
    return cost


def log_from_response(*, agent: str, model: str, response, extra: Optional[str] = None) -> float:
    """
    Convenience wrapper: extract token counts from a LangChain AIMessage
    (which carries usage_metadata) and call log_call().

    Falls back gracefully if usage info is absent.
    """
    usage = getattr(response, "usage_metadata", None)
    if usage:
        input_tokens  = usage.get("input_tokens",  0)
        output_tokens = usage.get("output_tokens", 0)
    else:
        # LangChain < 0.2 stores usage in response_metadata
        rm = getattr(response, "response_metadata", {})
        token_usage = rm.get("token_usage", {})
        input_tokens  = token_usage.get("prompt_tokens",     0)
        output_tokens = token_usage.get("completion_tokens", 0)

    return log_call(
        agent=agent,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        extra=extra,
    )


def session_summary() -> str:
    """Return a formatted session cost summary string."""
    return (
        f"📊 Session total: {_session.call_count} calls | "
        f"in={_session.total_input} out={_session.total_output} tokens | "
        f"est. cost=${_session.total_cost_usd:.4f}"
    )
