"""Math Tools — SymPy-based tools for Agent Tool Use.

Provides deterministic, exact mathematical computation that the
Teacher Agent can invoke during its Reflection (thinking) phase
to *verify* its own reasoning before responding to the student.

This eliminates the #1 failure mode of LLM math tutors:
    «hallucinated calculations» — e.g. the LLM writes
    "∫ x² dx = x³/2 + C" instead of "x³/3 + C".

Architecture
------------
Each tool is a plain Python function decorated with metadata.
The Reflection engine calls these tools programmatically (not via
LangChain Tool schema) to keep things simple and fast.

Supported operations:
  • compute_derivative — exact symbolic differentiation
  • compute_integral   — definite & indefinite integration
  • solve_equation     — algebraic/transcendental equation solving
  • simplify_expr      — simplification & LaTeX rendering
  • evaluate_expr      — numeric evaluation at a point
"""

from __future__ import annotations

import logging
import sympy as sp
from sympy import (
    symbols, diff, integrate, simplify, latex,
    sin, cos, tan, exp, log, sqrt, oo,
    solve, Eq, Rational, pi, E,
    limit as sp_limit,
)
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)
from typing import Optional

logger = logging.getLogger(__name__)

x, y, z, t, n = symbols("x y z t n")

# Safe parsing transformations
_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

# Allowed names for parse_expr (whitelist approach for security)
_LOCAL_DICT = {
    "x": x, "y": y, "z": z, "t": t, "n": n,
    "sin": sin, "cos": cos, "tan": tan,
    "exp": exp, "log": log, "ln": log,
    "sqrt": sqrt, "pi": pi, "e": E,
    "oo": oo, "inf": oo,
}


def _safe_parse(expr_str: str) -> sp.Expr:
    """Parse a string into a SymPy expression safely."""
    # Normalise common Vietnamese/LaTeX conventions
    expr_str = (
        expr_str
        .replace("^", "**")
        .replace("{", "(")
        .replace("}", ")")
        .replace("\\cdot", "*")
        .replace("\\times", "*")
        .replace("\\frac", "")
        .replace("\\left", "")
        .replace("\\right", "")
        .replace("\\", "")
        .strip()
    )
    return parse_expr(
        expr_str,
        local_dict=_LOCAL_DICT,
        transformations=_TRANSFORMS,
    )


# ── Tool Functions ───────────────────────────────────────────────────────────

def compute_derivative(expr_str: str, var: str = "x", order: int = 1) -> dict:
    """Compute the symbolic derivative of an expression.

    Returns
    -------
    dict with keys: success, result_latex, result_sympy, error
    """
    try:
        expr = _safe_parse(expr_str)
        v = symbols(var)
        result = diff(expr, v, order)
        result = simplify(result)
        return {
            "success": True,
            "result_latex": f"${latex(result)}$",
            "result_sympy": str(result),
            "input": expr_str,
            "operation": f"d^{order}/d{var}^{order}" if order > 1 else f"d/d{var}",
        }
    except Exception as e:
        logger.warning("Tool compute_derivative failed: %s", e)
        return {"success": False, "error": str(e)}


def compute_integral(
    expr_str: str,
    var: str = "x",
    lower: Optional[str] = None,
    upper: Optional[str] = None,
) -> dict:
    """Compute definite or indefinite integral.

    If lower/upper are provided → definite integral.
    Otherwise → indefinite integral (+ C).
    """
    try:
        expr = _safe_parse(expr_str)
        v = symbols(var)

        if lower is not None and upper is not None:
            a = _safe_parse(str(lower))
            b = _safe_parse(str(upper))
            result = integrate(expr, (v, a, b))
            op_str = f"∫[{lower},{upper}]"
        else:
            result = integrate(expr, v)
            op_str = "∫ ... dx"

        result = simplify(result)
        return {
            "success": True,
            "result_latex": f"${latex(result)}$",
            "result_sympy": str(result),
            "input": expr_str,
            "operation": op_str,
        }
    except Exception as e:
        logger.warning("Tool compute_integral failed: %s", e)
        return {"success": False, "error": str(e)}


def solve_equation(equation_str: str, var: str = "x") -> dict:
    """Solve an equation. Input can be 'expr = 0' or just 'expr' (assumed = 0)."""
    try:
        v = symbols(var)
        if "=" in equation_str:
            lhs_str, rhs_str = equation_str.split("=", 1)
            lhs = _safe_parse(lhs_str.strip())
            rhs = _safe_parse(rhs_str.strip())
            eq = Eq(lhs, rhs)
            solutions = solve(eq, v)
        else:
            expr = _safe_parse(equation_str)
            solutions = solve(expr, v)

        solutions_latex = [f"${latex(s)}$" for s in solutions]
        return {
            "success": True,
            "solutions": solutions_latex,
            "solutions_sympy": [str(s) for s in solutions],
            "count": len(solutions),
            "input": equation_str,
        }
    except Exception as e:
        logger.warning("Tool solve_equation failed: %s", e)
        return {"success": False, "error": str(e)}


def simplify_expression(expr_str: str) -> dict:
    """Simplify an expression and return LaTeX."""
    try:
        expr = _safe_parse(expr_str)
        result = simplify(expr)
        res = {
            "success": True,
            "result_latex": f"${latex(result)}$",
            "result_sympy": str(result),
        }
        if result.is_number:
            try:
                res["result_float"] = str(float(result.evalf()))
            except Exception:
                pass
        return res
    except Exception as e:
        return {"success": False, "error": str(e)}


def evaluate_at_point(expr_str: str, var: str = "x", value: str = "0") -> dict:
    """Evaluate expression at a specific point."""
    try:
        expr = _safe_parse(expr_str)
        v = symbols(var)
        val = _safe_parse(value)
        result = expr.subs(v, val)
        result = simplify(result)
        return {
            "success": True,
            "result_latex": f"${latex(result)}$",
            "result_sympy": str(result),
            "result_float": str(float(result)) if result.is_number else None,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Tool Registry ────────────────────────────────────────────────────────────
# Maps tool names → functions (used by the Reflection engine)

MATH_TOOLS = {
    "compute_derivative": compute_derivative,
    "compute_integral": compute_integral,
    "solve_equation": solve_equation,
    "simplify_expression": simplify_expression,
    "evaluate_at_point": evaluate_at_point,
}

TOOL_DESCRIPTIONS = {
    "compute_derivative": "Tính đạo hàm chính xác (symbolic). Params: expr_str, var='x', order=1",
    "compute_integral": "Tính tích phân (xác định/bất định). Params: expr_str, var='x', lower=None, upper=None",
    "solve_equation": "Giải phương trình. Params: equation_str, var='x'",
    "simplify_expression": "Rút gọn biểu thức. Params: expr_str",
    "evaluate_at_point": "Tính giá trị tại 1 điểm. Params: expr_str, var='x', value='0'",
}
