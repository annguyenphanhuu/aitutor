"""Exact-match formula registry backed by data/formulas.json."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

FORMULAS_PATH = Path(__file__).resolve().parents[2] / "data" / "formulas.json"


@lru_cache(maxsize=1)
def _formula_index() -> dict[str, dict]:
    """Load formulas once and index them by id."""
    if not FORMULAS_PATH.exists():
        return {}

    with FORMULAS_PATH.open("r", encoding="utf-8") as handle:
        formulas = json.load(handle)

    if not isinstance(formulas, list):
        return {}

    return {
        str(item["id"]): item
        for item in formulas
        if isinstance(item, dict) and item.get("id")
    }


def get_formula_by_id(formula_id: str) -> dict | None:
    """Return one formula by id, or None if it is unknown."""
    if not formula_id:
        return None
    return _formula_index().get(formula_id)


def get_formulas_by_ids(formula_ids: list[str] | tuple[str, ...] | None) -> list[dict]:
    """Return formulas in the same order as the requested ids."""
    if not formula_ids:
        return []

    index = _formula_index()
    formulas: list[dict] = []
    seen: set[str] = set()

    for formula_id in formula_ids:
        fid = str(formula_id).strip()
        if not fid or fid in seen:
            continue
        formula = index.get(fid)
        if formula:
            formulas.append(formula)
            seen.add(fid)

    return formulas


def list_formulas() -> list[dict]:
    """Return all known formulas in registry order."""
    return list(_formula_index().values())


def list_formula_ids() -> list[str]:
    """Return every known formula id."""
    return list(_formula_index().keys())


def clear_formula_cache() -> None:
    """Clear the cached registry, useful for tests after editing formulas.json."""
    _formula_index.cache_clear()
