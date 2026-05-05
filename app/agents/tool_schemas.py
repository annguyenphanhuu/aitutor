"""OpenAI Function Calling — tool schemas cho 5 SymPy math tools.

Định nghĩa theo OpenAI Chat Completions API tool format.
Được dùng bởi AgenticTeacherMixin để LLM tự quyết định khi nào
gọi tool thay vì heuristic regex scan.

Docs: https://platform.openai.com/docs/guides/function-calling
"""

from __future__ import annotations

# ── Tool definitions ───────────────────────────────────────────────────────────

OPENAI_MATH_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "compute_derivative",
            "description": (
                "Tính đạo hàm chính xác (symbolic) của một biểu thức bằng SymPy. "
                "Dùng khi cần kiểm chứng hoặc tính đạo hàm bậc 1 hoặc bậc n."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expr_str": {
                        "type": "string",
                        "description": (
                            "Biểu thức toán học dùng cú pháp Python/SymPy. "
                            "Ví dụ: 'x**3 + 2*x', 'sin(x)*exp(x)', 'log(x**2 + 1)'"
                        ),
                    },
                    "var": {
                        "type": "string",
                        "description": "Biến lấy đạo hàm theo. Mặc định 'x'.",
                        "default": "x",
                    },
                    "order": {
                        "type": "integer",
                        "description": "Bậc của đạo hàm. Mặc định 1 (đạo hàm bậc 1).",
                        "default": 1,
                    },
                },
                "required": ["expr_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_integral",
            "description": (
                "Tính tích phân xác định hoặc bất định bằng SymPy. "
                "Nếu có lower/upper → tích phân xác định (Newton–Leibniz). "
                "Nếu không có → tích phân bất định (+ C)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expr_str": {
                        "type": "string",
                        "description": "Biểu thức dưới dấu tích phân. Ví dụ: 'x**2', 'sin(x)*cos(x)'",
                    },
                    "var": {
                        "type": "string",
                        "description": "Biến tích phân. Mặc định 'x'.",
                        "default": "x",
                    },
                    "lower": {
                        "type": "string",
                        "description": "Cận dưới (cho tích phân xác định). Ví dụ: '0', 'pi/2'",
                    },
                    "upper": {
                        "type": "string",
                        "description": "Cận trên (cho tích phân xác định). Ví dụ: '1', 'pi'",
                    },
                },
                "required": ["expr_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "solve_equation",
            "description": (
                "Giải phương trình đại số bằng SymPy. "
                "Hỗ trợ dạng 'f(x) = g(x)' hoặc 'f(x)' (mặc định = 0). "
                "Trả về danh sách nghiệm chính xác."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "equation_str": {
                        "type": "string",
                        "description": (
                            "Phương trình cần giải. "
                            "Ví dụ: 'x**2 - 5*x + 6 = 0', 'sin(x) = 0.5', 'x**3 - 3*x'"
                        ),
                    },
                    "var": {
                        "type": "string",
                        "description": "Ẩn số cần tìm. Mặc định 'x'.",
                        "default": "x",
                    },
                },
                "required": ["equation_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "simplify_expression",
            "description": (
                "Rút gọn / đơn giản hóa một biểu thức toán học bằng SymPy. "
                "Trả về dạng LaTeX. Dùng khi cần kiểm tra kết quả đã ở dạng đơn giản nhất chưa."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expr_str": {
                        "type": "string",
                        "description": "Biểu thức cần rút gọn. Ví dụ: 'sin(x)**2 + cos(x)**2', '(x**2-1)/(x-1)'",
                    },
                },
                "required": ["expr_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_at_point",
            "description": (
                "Tính giá trị của biểu thức tại một điểm cụ thể. "
                "Dùng khi cần kiểm tra giá trị hàm số, đạo hàm tại điểm x = a."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expr_str": {
                        "type": "string",
                        "description": "Biểu thức cần tính. Ví dụ: 'x**2 + 2*x + 1'",
                    },
                    "var": {
                        "type": "string",
                        "description": "Tên biến. Mặc định 'x'.",
                        "default": "x",
                    },
                    "value": {
                        "type": "string",
                        "description": "Giá trị thay vào biến. Ví dụ: '0', '1', 'pi/2', '-1'",
                        "default": "0",
                    },
                },
                "required": ["expr_str", "value"],
            },
        },
    },
]

# Mapping từ tool name → function (import lazy để tránh circular)
def _get_math_tools() -> dict:
    from app.agents.tools import MATH_TOOLS  # noqa: PLC0415
    return MATH_TOOLS

TOOL_NAMES = {t["function"]["name"] for t in OPENAI_MATH_TOOLS}
