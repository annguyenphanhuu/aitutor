"""Small model-compatibility helpers shared by LLM clients."""

from __future__ import annotations


_REASONING_MODEL_MARKERS = ("o1", "o3", "o4")


def is_reasoning_model(model: str) -> bool:
    """Return whether a model uses the fixed reasoning-model temperature."""
    normalized = (model or "").casefold()
    return any(marker in normalized for marker in _REASONING_MODEL_MARKERS)


def compatible_temperature(model: str, standard_temperature: float) -> float:
    """Use temperature 1 for reasoning models, otherwise the requested value."""
    return 1.0 if is_reasoning_model(model) else standard_temperature
