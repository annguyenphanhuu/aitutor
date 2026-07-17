"""Small model-compatibility helpers shared by LLM clients."""

from __future__ import annotations

import re

# Match o1/o3/o4 as a model-name prefix (optionally after a provider path
# like "openai/o4-mini"), not as an arbitrary substring — tránh match nhầm
# các tên model chỉ tình cờ chứa "o4".
_REASONING_MODEL_RE = re.compile(r"(?:^|/)o[134](?:-|$)")


def is_reasoning_model(model: str) -> bool:
    """Return whether a model uses the fixed reasoning-model temperature."""
    normalized = (model or "").casefold()
    return bool(_REASONING_MODEL_RE.search(normalized))


def compatible_temperature(model: str, standard_temperature: float) -> float:
    """Use temperature 1 for reasoning models, otherwise the requested value."""
    return 1.0 if is_reasoning_model(model) else standard_temperature
