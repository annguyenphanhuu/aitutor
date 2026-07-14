"""Math Tools — SymPy-based tools for Agent Tool Use (Function Calling).

Provides deterministic, exact mathematical computation exposed as
LangChain @tool objects so the Solver LLM can call them DURING reasoning
(not after-the-fact). This is the standard AI-Agent pattern:
  LLM reasons → needs a calculation → calls tool → gets exact result → continues.

This eliminates the #1 failure mode of LLM math tutors:
    «hallucinated calculations» — e.g. the LLM writes
    "∫ x² dx = x³/2 + C" instead of "x³/3 + C".

Architecture
------------
Each tool is a Python function decorated with @tool (LangChain).
The Solver Agent's LLM has these tools bound via llm.bind_tools(MATH_TOOL_LIST),
enabling native function-calling (OpenAI tool_choice protocol).

The legacy MATH_TOOLS dict is kept for backward compatibility only.

Supported operations:
  • compute_derivative   — exact symbolic differentiation
  • compute_integral     — definite & indefinite integration
  • solve_equation       — algebraic/transcendental equation solving
  • solve_inequality     — inequality solving (e.g. 2^(n/20) > 400)
  • simplify_expression  — simplification & LaTeX rendering
  • evaluate_at_point    — numeric evaluation at a point
"""

from __future__ import annotations

import logging
import sympy as sp
from sympy import (
    symbols, diff, integrate, simplify, latex,
    sin, cos, tan, exp, log, sqrt, oo,
    solve, solve_univariate_inequality, Eq, pi, E,
)
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

x, y, z, t, n = symbols("x y z t n", real=True)

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


def _symbol_for(var: str) -> sp.Symbol:
    """Return the exact Symbol instance used by ``_safe_parse``."""
    known = _LOCAL_DICT.get(var)
    return known if isinstance(known, sp.Symbol) else symbols(var, real=True)


# ── Tool Functions ────────────────────────────────────────────────────────────
# Public functions return structured dictionaries. LangChain wrappers are built
# separately at the bottom so direct callers, Reflection and Function Calling
# all share the same stable contract.

def compute_derivative(expr_str: str, var: str = "x", order: int = 1) -> dict:
    """Tính đạo hàm bậc N của một biểu thức toán học bằng SymPy (chính xác tuyệt đối).
    Dùng khi bài toán yêu cầu tính f'(x), f''(x), hoặc đạo hàm bất kỳ.

    Args:
        expr_str: Biểu thức cần tính đạo hàm (ví dụ: 'x**3 + 2*x', '4*t**3 - 72*t**2 + 288*t').
        var: Biến đạo hàm theo (mặc định 'x').
        order: Bậc đạo hàm (mặc định 1).

    Returns:
        Kết quả đạo hàm dưới dạng LaTeX và sympy string.
    """
    try:
        expr = _safe_parse(expr_str)
        v = _symbol_for(var)
        result = diff(expr, v, order)
        result = simplify(result)
        payload = {
            "success": True,
            "input": expr_str,
            "operation": "derivative",
            "result_sympy": str(result),
            "result_latex": f"${latex(result)}$",
        }
        if result.is_number:
            try:
                payload["result_float"] = str(float(result.evalf()))
            except Exception:
                pass
        return payload
    except Exception as e:
        logger.warning("Tool compute_derivative failed: %s", e)
        return {"success": False, "input": expr_str, "error": str(e)}


def compute_integral(expr_str: str, var: str = "x", lower: str = "", upper: str = "") -> dict:
    """Tính tích phân xác định hoặc bất định bằng SymPy (chính xác tuyệt đối).
    Dùng khi bài toán yêu cầu tính ∫f(x)dx hoặc ∫[a,b]f(x)dx.

    Args:
        expr_str: Biểu thức dưới dấu tích phân (ví dụ: 'x**2', 'sin(x)*cos(x)').
        var: Biến tích phân (mặc định 'x').
        lower: Cận dưới (để trống nếu là tích phân bất định).
        upper: Cận trên (để trống nếu là tích phân bất định).

    Returns:
        Kết quả tích phân dưới dạng LaTeX và giá trị thập phân (nếu có).
    """
    try:
        expr = _safe_parse(expr_str)
        v = _symbol_for(var)

        if lower and upper:
            a = _safe_parse(str(lower))
            b = _safe_parse(str(upper))
            result = integrate(expr, (v, a, b))
            operation = "definite_integral"
        else:
            result = integrate(expr, v)
            operation = "indefinite_integral"

        result = simplify(result)
        payload = {
            "success": True,
            "input": expr_str,
            "operation": operation,
            "result_sympy": str(result),
            "result_latex": f"${latex(result)}$",
        }
        if result.is_number:
            try:
                payload["result_float"] = str(float(result.evalf()))
            except Exception:
                pass
        return payload
    except Exception as e:
        logger.warning("Tool compute_integral failed: %s", e)
        return {"success": False, "input": expr_str, "error": str(e)}


def solve_equation(equation_str: str, var: str = "x") -> dict:
    """Giải phương trình toán học bằng SymPy (chính xác tuyệt đối).
    Dùng khi cần tìm giá trị của biến thỏa mãn phương trình.
    Input có thể là 'f(x) = g(x)' hoặc chỉ 'f(x)' (ngầm = 0).

    Args:
        equation_str: Phương trình cần giải (ví dụ: '2**n = 400', 'x**2 - 5*x + 6 = 0').
        var: Tên biến cần giải (mặc định 'x', dùng 'n' nếu bài dùng n, 't' nếu dùng t).

    Returns:
        Tập nghiệm chính xác và giá trị thập phân.
    """
    try:
        v = _symbol_for(var)
        if "=" in equation_str:
            lhs_str, rhs_str = equation_str.split("=", 1)
            lhs = _safe_parse(lhs_str.strip())
            rhs = _safe_parse(rhs_str.strip())
            eq = Eq(lhs, rhs)
            solutions = solve(eq, v)
        else:
            expr = _safe_parse(equation_str)
            solutions = solve(expr, v)

        return {
            "success": True,
            "input": equation_str,
            "operation": "solve_equation",
            "solutions_sympy": [str(solution) for solution in solutions],
            "solutions": [f"${latex(solution)}$" for solution in solutions],
            "count": len(solutions),
        }
    except Exception as e:
        logger.warning("Tool solve_equation failed: %s", e)
        return {"success": False, "input": equation_str, "error": str(e)}


def solve_inequality(inequality_str: str, var: str = "n") -> dict:
    """Giải bất phương trình toán học bằng SymPy (chính xác tuyệt đối).
    PHẢI dùng khi bài hỏi 'sau bao nhiêu giờ/phút/ngày' hoặc
    bất kỳ bài toán nào có dạng f(n) > C hoặc f(n) < C, f(n) >= C.
    Đặc biệt cần thiết cho bài toán tăng trưởng dân số, vi khuẩn, lãi kép.

    Args:
        inequality_str: Bất phương trình cần giải (ví dụ: '5 * 2**(n/20) > 2000', '2**n > 400').
        var: Tên biến cần giải (mặc định 'n').

    Returns:
        Nghiệm của bất phương trình và giá trị biên thập phân.
    """
    try:
        import sympy as sp
        from sympy import Symbol, N
        v = Symbol(var, real=True)
        # Parse with the variable substituted
        local = dict(_LOCAL_DICT)
        local[var] = v
        expr_str = (
            inequality_str
            .replace("^", "**")
            .replace("{", "(")
            .replace("}", ")")
            .replace("\\cdot", "*")
            .replace("\\times", "*")
            .replace("\\", "")
            .strip()
        )
        # Map inequality operator to SymPy
        for op, sym_rel in [(">=", sp.Ge), ("<=", sp.Le), (">", sp.Gt), ("<", sp.Lt)]:
            if op in expr_str:
                lhs_s, rhs_s = expr_str.split(op, 1)
                lhs = parse_expr(lhs_s.strip(), local_dict=local, transformations=_TRANSFORMS)
                rhs = parse_expr(rhs_s.strip(), local_dict=local, transformations=_TRANSFORMS)
                rel = sym_rel(lhs, rhs)
                solution = solve_univariate_inequality(rel, v, relational=False)
                # Get the critical bound as float
                boundary_floats = []
                for atom in solution.boundary:
                    try:
                        boundary_floats.append((latex(atom), float(N(atom))))
                    except Exception:
                        pass
                return {
                    "success": True,
                    "input": inequality_str,
                    "operation": "solve_inequality",
                    "result_sympy": str(solution),
                    "result_latex": f"${latex(solution)}$",
                    "boundaries": [
                        {"latex": boundary_latex, "float": boundary_float}
                        for boundary_latex, boundary_float in boundary_floats
                    ],
                    "note": (
                        "Nếu bài toán có chu kỳ rời rạc, làm tròn biên lên "
                        "đến bội số nguyên gần nhất của chu kỳ."
                    ),
                }
        return {
            "success": False,
            "input": inequality_str,
            "error": f"Không nhận dạng được dấu bất phương trình trong: {inequality_str}",
        }
    except Exception as e:
        logger.warning("Tool solve_inequality failed: %s", e)
        return {"success": False, "input": inequality_str, "error": str(e)}


def simplify_expression(expr_str: str) -> dict:
    """Rút gọn / tính toán một biểu thức toán học bằng SymPy.
    Dùng để kiểm tra hoặc tính giá trị cuối cùng của một biểu thức phức tạp.

    Args:
        expr_str: Biểu thức cần rút gọn (ví dụ: '9/4 - 11/12 + 11/12 - 9/4', '0.6*0.65 + 0.4*0.25').

    Returns:
        Kết quả rút gọn dưới dạng LaTeX và giá trị thập phân.
    """
    try:
        expr = _safe_parse(expr_str)
        result = simplify(expr)
        payload = {
            "success": True,
            "input": expr_str,
            "operation": "simplify",
            "result_sympy": str(result),
            "result_latex": f"${latex(result)}$",
        }
        if result.is_number:
            try:
                payload["result_float"] = str(float(result.evalf()))
            except Exception:
                pass
        return payload
    except Exception as e:
        return {"success": False, "input": expr_str, "error": str(e)}


def evaluate_at_point(expr_str: str, var: str = "x", value: str = "0") -> dict:
    """Tính giá trị của một biểu thức tại một điểm cụ thể bằng SymPy.
    Dùng khi cần tính f(a) với giá trị cụ thể của biến.

    Args:
        expr_str: Biểu thức cần tính (ví dụ: '4*t**3 - 72*t**2 + 288*t').
        var: Tên biến (mặc định 'x').
        value: Giá trị thay vào (ví dụ: '6', '2.5', 'pi/2').

    Returns:
        Giá trị chính xác và giá trị thập phân.
    """
    try:
        expr = _safe_parse(expr_str)
        v = _symbol_for(var)
        val = _safe_parse(value)
        result = expr.subs(v, val)
        result = simplify(result)
        payload = {
            "success": True,
            "input": expr_str,
            "operation": "evaluate_at_point",
            "result_sympy": str(result),
            "result_latex": f"${latex(result)}$",
        }
        if result.is_number:
            try:
                payload["result_float"] = str(float(result.evalf()))
            except Exception:
                pass
        return payload
    except Exception as e:
        return {"success": False, "input": expr_str, "error": str(e)}


# ── Tool Collections ─────────────────────────────────────────────────────────

# Canonical registry of plain functions. Direct Python callers receive
# dictionaries; LangChain/OpenAI wrappers are derived from it below.
MATH_TOOLS = {
    "compute_derivative": compute_derivative,
    "compute_integral": compute_integral,
    "solve_equation": solve_equation,
    "solve_inequality": solve_inequality,
    "simplify_expression": simplify_expression,
    "evaluate_at_point": evaluate_at_point,
}

MATH_TOOL_LIST = [tool(fn) for fn in MATH_TOOLS.values()]

TOOL_DESCRIPTIONS = {
    name: (fn.__doc__ or name).strip().splitlines()[0]
    for name, fn in MATH_TOOLS.items()
}
