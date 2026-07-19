"""Kiểm chứng đáp án học sinh bằng SymPy — deterministic, không LLM.

Verdict đúng/sai của bài chấm lấy từ đây khi kiểm chứng được; LLM chỉ
đóng vai trò trích xuất và viết nhận xét sư phạm. Mọi trường hợp không
parse/kiểm chứng được đều trả ``verified=False`` (hạ độ tin, không bao
giờ chặn response).
"""

from __future__ import annotations

import logging

import sympy as sp

from app.agents.contracts import GradingExtraction, VerifierResult
from app.agents.tools import _safe_parse, _symbol_for

logger = logging.getLogger(__name__)

_NUMERIC_TOL = 1e-9
# Điểm mẫu để đối chiếu số khi simplify không kết luận được
_SPOT_POINTS = (0.3, 1.7, 2.9)


def _strip_constant(expr: sp.Expr, var: sp.Symbol) -> sp.Expr:
    """Bỏ hằng số tích phân (+C, +K...) — các hạng tử cộng không chứa biến."""
    terms = sp.Add.make_args(sp.expand(expr))
    kept = [term for term in terms if var in term.free_symbols]
    return sp.Add(*kept) if kept else sp.Integer(0)


def _is_zero(expr: sp.Expr, var: sp.Symbol) -> bool | None:
    """Ba tầng kiểm tra expr == 0: simplify bị chặn ratio → equals → numeric.

    Trả None khi không kết luận được (SymPy bó tay và numeric cũng lỗi).
    """
    try:
        if sp.simplify(expr, ratio=1.7) == 0:
            return True
    except Exception:
        pass
    try:
        result = expr.equals(0)
        if result is not None:
            return bool(result)
    except Exception:
        pass
    try:
        for point in _SPOT_POINTS:
            value = complex(expr.subs(var, point))
            if abs(value) > _NUMERIC_TOL:
                return False
        return True
    except Exception:
        return None


def verify_antiderivative(candidate: str, integrand: str, var: str = "x") -> VerifierResult:
    """Kiểm tra F(x) có phải nguyên hàm của f(x): simplify(F' − f) == 0."""
    symbol = _symbol_for(var)
    F = _strip_constant(_safe_parse(candidate), symbol)
    f = _safe_parse(integrand)
    is_zero = _is_zero(sp.diff(F, symbol) - f, symbol)
    if is_zero is None:
        return VerifierResult(
            verified=False, verdict="unknown", method="antiderivative_roundtrip",
            details=f"Không kết luận được d/d{var}({candidate}) − ({integrand})",
        )
    return VerifierResult(
        verified=True,
        verdict="correct" if is_zero else "incorrect",
        method="antiderivative_roundtrip",
        details=f"d/d{var}({candidate}) so với {integrand}",
    )


def verify_derivative(candidate: str, original: str, var: str = "x") -> VerifierResult:
    """Kiểm tra g(x) có phải đạo hàm của f(x): simplify(f' − g) == 0."""
    symbol = _symbol_for(var)
    g = _safe_parse(candidate)
    f = _safe_parse(original)
    is_zero = _is_zero(sp.diff(f, symbol) - g, symbol)
    if is_zero is None:
        return VerifierResult(
            verified=False, verdict="unknown", method="derivative_check",
            details=f"Không kết luận được ({original})' − ({candidate})",
        )
    return VerifierResult(
        verified=True,
        verdict="correct" if is_zero else "incorrect",
        method="derivative_check",
        details=f"({original})' so với {candidate}",
    )


def verify_equation_solutions(candidate: str, equation: str, var: str = "x") -> VerifierResult:
    """So khớp tập nghiệm học sinh đưa ra với tập nghiệm SymPy giải được."""
    symbol = _symbol_for(var)
    if "=" in equation:
        left, right = equation.split("=", 1)
        eq = sp.Eq(_safe_parse(left), _safe_parse(right))
    else:
        eq = sp.Eq(_safe_parse(equation), 0)

    true_solutions = sp.solve(eq, symbol)
    student_solutions = [_safe_parse(part) for part in candidate.split(";") if part.strip()]

    def _matches(a: sp.Expr, b: sp.Expr) -> bool:
        result = sp.simplify(a - b) == 0
        return bool(result)

    missing = [s for s in true_solutions if not any(_matches(s, c) for c in student_solutions)]
    extra = [c for c in student_solutions if not any(_matches(c, s) for s in true_solutions)]
    correct = not missing and not extra
    return VerifierResult(
        verified=True,
        verdict="correct" if correct else "incorrect",
        method="equation_solution_set",
        details=f"nghiệm đúng={true_solutions}, học sinh={student_solutions}",
    )


def verify_numeric(candidate: str, expected_expr: str) -> VerifierResult:
    """So sánh hai giá trị số với sai số tuyệt đối 1e-9."""
    a = complex(sp.N(_safe_parse(candidate)))
    b = complex(sp.N(_safe_parse(expected_expr)))
    return VerifierResult(
        verified=True,
        verdict="correct" if abs(a - b) <= _NUMERIC_TOL else "incorrect",
        method="numeric_equality",
        details=f"{candidate} so với {expected_expr}",
    )


def verify(extraction: GradingExtraction) -> VerifierResult:
    """Dispatch theo task_kind. Mọi exception → verified=False (không chặn chấm)."""
    if not extraction.gradable or not extraction.candidate_expr or not extraction.problem_expr:
        return VerifierResult(verified=False, verdict="unknown", method="not_applicable")
    try:
        if extraction.task_kind == "antiderivative":
            return verify_antiderivative(extraction.candidate_expr, extraction.problem_expr)
        if extraction.task_kind == "derivative":
            return verify_derivative(extraction.candidate_expr, extraction.problem_expr)
        if extraction.task_kind == "equation":
            return verify_equation_solutions(extraction.candidate_expr, extraction.problem_expr)
        if extraction.task_kind == "numeric":
            return verify_numeric(extraction.candidate_expr, extraction.problem_expr)
        return VerifierResult(verified=False, verdict="unknown", method="unsupported_kind")
    except Exception as exc:  # parse lỗi, solve treo, input rác...
        logger.warning("Verifier không kiểm chứng được (%s): %s", extraction.task_kind, exc)
        return VerifierResult(
            verified=False, verdict="unknown", method="parse_error", details=str(exc)
        )
