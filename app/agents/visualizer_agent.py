"""Visualizer Agent — generates chart data (Plotly) and Desmos URLs for math visualization."""

import re
import numpy as np

try:
    import sympy as sp
    from sympy import symbols, lambdify, latex
    from sympy.parsing.sympy_parser import (
        parse_expr,
        standard_transformations,
        implicit_multiplication_application,
        convert_xor,
    )
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False


x = symbols("x") if SYMPY_AVAILABLE else None

# Common transformations for parsing user input
TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


class VisualizerAgent:
    """Agent that generates visualization data for math concepts."""

    def generate_function_plot(self, expression_str: str) -> dict:
        """
        Parse a math expression and generate Plotly-compatible trace data.

        Args:
            expression_str: e.g. "x^3 - 3x + 1", "sin(x)", "e^x"

        Returns:
            {type, data, latex_caption} for frontend rendering
        """
        if not SYMPY_AVAILABLE:
            return self._error("SymPy không khả dụng.")

        try:
            # Clean up the expression string
            expr_clean = self._normalize_expr(expression_str)
            expr = parse_expr(expr_clean, transformations=TRANSFORMS, local_dict={"x": x, "e": sp.E})

            # Generate numeric data
            f_numeric = lambdify(x, expr, modules=["numpy"])
            x_vals = np.linspace(-10, 10, 500)

            try:
                y_vals = f_numeric(x_vals)
            except Exception:
                y_vals = np.array([float(expr.subs(x, xv)) for xv in x_vals])

            # Guard: lambdify may return an object array when expression
            # contains unsupported symbols — np.isfinite would crash on it.
            if not np.issubdtype(np.asarray(y_vals).dtype, np.number):
                # Fallback: evaluate point-by-point with SymPy
                try:
                    y_vals = np.array(
                        [float(expr.subs(x, float(xv))) for xv in x_vals],
                        dtype=float,
                    )
                except Exception:
                    return self._error(
                        "Không thể tính giá trị hàm số. "
                        "Em hãy ghi rõ biểu thức, ví dụ: y = 2x*sin(x) + x^2*cos(x)"
                    )

            # Handle infinities and NaN
            y_vals = np.where(np.isfinite(y_vals), y_vals, None)

            # Auto-scale y-axis
            finite_y = [y for y in y_vals if y is not None]
            if finite_y:
                y_min = max(min(finite_y), -50)
                y_max = min(max(finite_y), 50)
            else:
                y_min, y_max = -10, 10

            expr_latex = latex(expr)

            return {
                "vis_type": "plot",
                "data": {
                    "traces": [{
                        "x": x_vals.tolist(),
                        "y": [float(y) if y is not None else None for y in y_vals],
                        "type": "scatter",
                        "mode": "lines",
                        "name": f"y = {expr_latex}",
                        "line": {"color": "#6366f1", "width": 3},
                    }],
                    "layout": {
                        "title": f"Đồ thị hàm số $y = {expr_latex}$",
                        "xaxis": {"title": "x", "zeroline": True, "zerolinecolor": "#888", "gridcolor": "#e5e7eb"},
                        "yaxis": {"title": "y", "zeroline": True, "zerolinecolor": "#888", "gridcolor": "#e5e7eb",
                                  "range": [y_min * 1.1, y_max * 1.1]},
                        "plot_bgcolor": "#f9fafb",
                        "paper_bgcolor": "transparent",
                        "font": {"family": "Inter, sans-serif"},
                        "margin": {"t": 50, "b": 50, "l": 50, "r": 30},
                    },
                },
                "latex_caption": f"y = {expr_latex}",
            }

        except Exception as e:
            return self._error(f"Không thể vẽ biểu thức: {str(e)}")

    def generate_desmos_url(self, expression_str: str) -> dict:
        """Generate a Desmos calculator embed URL."""
        # Clean expression for Desmos format
        expr_clean = self._normalize_expr(expression_str)
        # Desmos uses its own LaTeX-like expression format
        desmos_expr = expr_clean.replace("**", "^").replace("*", "")

        calc_url = "https://www.desmos.com/calculator"

        return {
            "vis_type": "desmos",
            "data": {
                "url": calc_url,
                "expression": desmos_expr,
                "embed_html": (
                    f'<iframe src="{calc_url}" '
                    f'width="100%" height="400" '
                    f'style="border:1px solid #ccc;border-radius:8px;" '
                    f'frameborder="0"></iframe>'
                ),
            },
            "latex_caption": expression_str,
        }

    def generate_derivative_visualization(self, expression_str: str) -> dict:
        """Plot a function alongside its derivative."""
        if not SYMPY_AVAILABLE:
            return self._error("SymPy không khả dụng.")

        try:
            expr_clean = self._normalize_expr(expression_str)
            expr = parse_expr(expr_clean, transformations=TRANSFORMS, local_dict={"x": x, "e": sp.E})
            deriv = sp.diff(expr, x)

            f_num = lambdify(x, expr, modules=["numpy"])
            fp_num = lambdify(x, deriv, modules=["numpy"])

            x_vals = np.linspace(-10, 10, 500)
            y_vals = np.where(np.isfinite(f_num(x_vals)), f_num(x_vals), None)
            yp_vals = np.where(np.isfinite(fp_num(x_vals)), fp_num(x_vals), None)

            return {
                "vis_type": "plot",
                "data": {
                    "traces": [
                        {
                            "x": x_vals.tolist(),
                            "y": [float(y) if y is not None else None for y in y_vals],
                            "type": "scatter", "mode": "lines",
                            "name": f"f(x) = {latex(expr)}",
                            "line": {"color": "#6366f1", "width": 3},
                        },
                        {
                            "x": x_vals.tolist(),
                            "y": [float(y) if y is not None else None for y in yp_vals],
                            "type": "scatter", "mode": "lines",
                            "name": f"f'(x) = {latex(deriv)}",
                            "line": {"color": "#f97316", "width": 2, "dash": "dash"},
                        },
                    ],
                    "layout": {
                        "title": f"Hàm số và đạo hàm của $f(x) = {latex(expr)}$",
                        "xaxis": {"title": "x", "zeroline": True, "gridcolor": "#e5e7eb"},
                        "yaxis": {"title": "y", "zeroline": True, "gridcolor": "#e5e7eb",
                                  "range": [-20, 20]},
                        "plot_bgcolor": "#f9fafb",
                        "paper_bgcolor": "transparent",
                        "font": {"family": "Inter, sans-serif"},
                        "legend": {"x": 0, "y": 1.15, "orientation": "h"},
                        "margin": {"t": 60, "b": 50, "l": 50, "r": 30},
                    },
                },
                "latex_caption": f"f(x) = {latex(expr)},\\quad f'(x) = {latex(deriv)}",
            }

        except Exception as e:
            return self._error(f"Không thể vẽ: {str(e)}")

    def _normalize_expr(self, s: str) -> str:
        """Normalize user input to parseable math expression."""
        s = s.strip()
        # Remove y= or f(x)= prefix
        s = re.sub(r"^[yf]\s*\(?\s*x?\s*\)?\s*=\s*", "", s)
        # Common substitutions
        s = s.replace("^", "**")
        s = s.replace("√", "sqrt")
        s = s.replace("ln", "log")
        # e^x -> exp(x): wrap exponent properly
        s = re.sub(r'\be\*\*\(([^)]+)\)', r'exp(\1)', s)  # e**(x) -> exp(x)
        s = re.sub(r'\be\*\*(\w+)', r'exp(\1)', s)          # e**x -> exp(x)
        return s

    def _error(self, msg: str) -> dict:
        return {
            "vis_type": "error",
            "data": {"message": msg},
            "latex_caption": None,
        }
