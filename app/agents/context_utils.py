"""Shared curriculum-context formatting for teaching agents."""

from __future__ import annotations

from collections.abc import Iterable

from app.knowledge_tracing.skill_graph import SKILLS
from app.rag.formula_registry import get_formulas_by_ids


def normalize_skill_ids(
    skill_ids: Iterable[str] | None = None,
    primary_skill_id: str | None = None,
) -> list[str]:
    """Return unique, non-empty skill ids with the primary skill first."""
    normalized: list[str] = []
    for skill_id in skill_ids or []:
        if skill_id and skill_id not in normalized:
            normalized.append(skill_id)
    if primary_skill_id and primary_skill_id not in normalized:
        normalized.insert(0, primary_skill_id)
    return normalized


def format_skill_names(skill_ids: Iterable[str] | None) -> str:
    """Format skill ids and Vietnamese display names for prompts."""
    normalized = normalize_skill_ids(skill_ids)
    if not normalized:
        return "chua xac dinh"
    return ", ".join(
        f"{SKILLS.get(skill_id, {}).get('name', skill_id)} ({skill_id})"
        for skill_id in normalized
    )


def format_formulas(formula_ids: Iterable[str] | None) -> str:
    """Resolve formula ids and format their canonical content for prompts."""
    formulas = get_formulas_by_ids(list(formula_ids or []))
    if not formulas:
        return "Khong co cong thuc trong tam duoc gan metadata."
    return "\n\n".join(formula.get("content", "") for formula in formulas)
