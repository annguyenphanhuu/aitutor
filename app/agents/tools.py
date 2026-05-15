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
    solve, solve_univariate_inequality, Eq, Rational, pi, E,
    limit as sp_limit,
)
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)
from langchain_core.tools import tool
from typing import Optional

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


# ── Tool Functions (decorated with @tool for LangChain function-calling) ────────

@tool
def compute_derivative(expr_str: str, var: str = "x", order: int = 1) -> str:
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
        v = symbols(var)
        result = diff(expr, v, order)
        result = simplify(result)
        result_latex = latex(result)
        result_float = None
        if result.is_number:
            try:
                result_float = float(result.evalf())
            except Exception:
                pass
        return (
            f"SymPy result: d^{order}/d{var}^{order}({expr_str}) = {result_latex}"
            + (f" ~ {result_float:.6f}" if result_float is not None else "")
        )
    except Exception as e:
        logger.warning("Tool compute_derivative failed: %s", e)
        return f"ERROR: {e}"


@tool
def compute_integral(expr_str: str, var: str = "x", lower: str = "", upper: str = "") -> str:
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
        v = symbols(var)

        if lower and upper:
            a = _safe_parse(str(lower))
            b = _safe_parse(str(upper))
            result = integrate(expr, (v, a, b))
            op_str = f"∫[{lower},{upper}] {expr_str} d{var}"
        else:
            result = integrate(expr, v)
            op_str = f"∫ {expr_str} d{var}"

        result = simplify(result)
        result_latex = latex(result)
        result_float = None
        if result.is_number:
            try:
                result_float = float(result.evalf())
            except Exception:
                pass
        return (
            f"SymPy result: {op_str} = {result_latex}"
            + (f" ~ {result_float:.6f}" if result_float is not None else "")
        )
    except Exception as e:
        logger.warning("Tool compute_integral failed: %s", e)
        return f"ERROR: {e}"


@tool
def solve_equation(equation_str: str, var: str = "x") -> str:
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

        if not solutions:
            return f"SymPy result: Phương trình '{equation_str}' vô nghiệm."

        parts = []
        for s in solutions:
            s_latex = latex(s)
            try:
                s_float = float(s.evalf())
                parts.append(f"{s_latex} ~ {s_float:.6f}")
            except Exception:
                parts.append(s_latex)

        return f"SymPy result: {var} in {{{', '.join(parts)}}}"
    except Exception as e:
        logger.warning("Tool solve_equation failed: %s", e)
        return f"ERROR: {e}"


@tool
def solve_inequality(inequality_str: str, var: str = "n") -> str:
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
        from sympy import Symbol, log, N
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
                bound_str = "; ".join(
                    f"{la} ~ {fl:.6f}" for la, fl in boundary_floats
                )
                return (
                    f"SymPy result: Bpt '{inequality_str}' nghiem: {latex(solution)}"
                    + (f" | Bien: {bound_str}" if bound_str else "")
                    + "\nLUU Y: Neu bai toan co chu ky roi rac (vi du vi khuan phan bao moi 20 phut), "
                    "hay lam tron bien LEN den boi so nguyen gan nhat cua chu ky do."
                )
        return f"ERROR: Không nhận dạng được dấu bất phương trình trong: {inequality_str}"
    except Exception as e:
        logger.warning("Tool solve_inequality failed: %s", e)
        return f"ERROR: {e}"


@tool
def simplify_expression(expr_str: str) -> str:
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
        result_latex = latex(result)
        extra = ""
        if result.is_number:
            try:
                result_float = float(result.evalf())
                extra = f" ~ {result_float:.6f}"
            except Exception:
                pass
        return f"SymPy result: simplify({expr_str}) = {result_latex}{extra}"
    except Exception as e:
        return f"ERROR: {e}"


@tool
def evaluate_at_point(expr_str: str, var: str = "x", value: str = "0") -> str:
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
        v = symbols(var)
        val = _safe_parse(value)
        result = expr.subs(v, val)
        result = simplify(result)
        result_latex = latex(result)
        extra = ""
        if result.is_number:
            try:
                result_float = float(result.evalf())
                extra = f" ~ {result_float:.6f}"
            except Exception:
                pass
        return f"SymPy result: {expr_str} at {var}={value} = {result_latex}{extra}"
    except Exception as e:
        return f"ERROR: {e}"


# ── Tool Collections ─────────────────────────────────────────────────────────

# List of @tool objects for llm.bind_tools() — used by the Agentic Solver
MATH_TOOL_LIST = [
    compute_derivative,
    compute_integral,
    solve_equation,
    solve_inequality,
    simplify_expression,
    evaluate_at_point,
]

# Legacy dict registry — kept for backward compatibility with ReflectionEngine
MATH_TOOLS = {
    "compute_derivative":  compute_derivative.func if hasattr(compute_derivative, "func") else compute_derivative,
    "compute_integral":    compute_integral.func if hasattr(compute_integral, "func") else compute_integral,
    "solve_equation":      solve_equation.func if hasattr(solve_equation, "func") else solve_equation,
    "solve_inequality":    solve_inequality.func if hasattr(solve_inequality, "func") else solve_inequality,
    "simplify_expression": simplify_expression.func if hasattr(simplify_expression, "func") else simplify_expression,
    "evaluate_at_point":   evaluate_at_point.func if hasattr(evaluate_at_point, "func") else evaluate_at_point,
}

TOOL_DESCRIPTIONS = {
    "compute_derivative":  "Tính đạo hàm chính xác (symbolic). Params: expr_str, var='x', order=1",
    "compute_integral":    "Tính tích phân (xác định/bất định). Params: expr_str, var='x', lower='', upper=''",
    "solve_equation":      "Giải phương trình. Params: equation_str, var='x'",
    "solve_inequality":    "Giải bất phương trình (>, <, >=, <=). Params: inequality_str, var='n'",
    "simplify_expression": "Rút gọn biểu thức. Params: expr_str",
    "evaluate_at_point":   "Tính giá trị tại 1 điểm. Params: expr_str, var='x', value='0'",
}
