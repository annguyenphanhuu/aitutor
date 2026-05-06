"""Quiz Generator — creates math questions using SymPy for exact answers."""

import random
import sympy as sp
from sympy import (
    symbols, diff, integrate, simplify, latex, Rational,
    sin, cos, tan, exp, log, sqrt, oo,
    solve, Eq,
)


x = symbols("x")


# ── Base Generator ───────────────────────────────────────
class BaseQuizGenerator:
    """Base class for quiz generators. Subclasses implement _templates()."""

    skill_id: str = ""

    def generate(self, difficulty: int = 1) -> dict:
        """
        Generate a single quiz question.

        Returns: {question_latex, choices, correct_index, explanation, difficulty, skill_id, sympy_expr}
        """
        templates = self._templates(difficulty)
        template_fn = random.choice(templates)
        return template_fn()

    def _templates(self, difficulty: int) -> list:
        raise NotImplementedError

    @staticmethod
    def _make_mcq(correct_val, skill_id: str, question_text: str,
                  explanation: str, difficulty: int, sympy_expr: str = "") -> dict:
        """Build an MCQ dict with 4 choices (1 correct + 3 distractors)."""
        correct_latex = f"${latex(correct_val)}$"

        # Generate distractors by perturbing the correct answer
        distractors = set()
        attempts = 0
        while len(distractors) < 3 and attempts < 30:
            attempts += 1
            d = _perturb(correct_val)
            d_latex = f"${latex(d)}$"
            if d_latex != correct_latex and d_latex not in distractors:
                distractors.add(d_latex)

        # Fill remaining with simple numeric offsets if needed
        while len(distractors) < 3:
            filler = sp.Integer(random.randint(-5, 5))
            fl = f"${latex(filler)}$"
            if fl != correct_latex and fl not in distractors:
                distractors.add(fl)

        choices = [correct_latex] + list(distractors)
        random.shuffle(choices)
        correct_index = choices.index(correct_latex)

        return {
            "question_type": "mcq",
            "question_latex": question_text,
            "choices": choices,
            "correct_index": correct_index,
            "points": 0.25,
            "explanation": explanation,
            "difficulty": difficulty,
            "skill_id": skill_id,
            "skill_ids": [skill_id],
            "formula_ids": [],
            "sympy_expr": sympy_expr,
        }

    @staticmethod
    def _make_true_false(statements_data: list[tuple], skill_id: str,
                         question_text: str, explanation: str, difficulty: int,
                         sympy_expr: str = "") -> dict:
        """
        Build a True/False question with 4 statements.
        statements_data: [(text, is_correct_bool), ...] — exactly 4 items
        """
        statements = [
            {"text": text, "correct": correct}
            for text, correct in statements_data
        ]
        return {
            "question_type": "true_false",
            "question_latex": question_text,
            "statements": statements,
            "points": 1.0,  # max 1.0 for 4/4 correct
            "explanation": explanation,
            "difficulty": difficulty,
            "skill_id": skill_id,
            "skill_ids": [skill_id],
            "formula_ids": [],
            "sympy_expr": sympy_expr,
        }

    @staticmethod
    def _make_short_answer(correct_val, skill_id: str, question_text: str,
                           explanation: str, difficulty: int,
                           sympy_expr: str = "") -> dict:
        """Build a short-answer question where student types a numeric answer."""
        # Convert SymPy to string answer
        try:
            answer_str = str(float(correct_val))
            # Clean up: remove trailing .0 for integers
            if answer_str.endswith('.0'):
                answer_str = answer_str[:-2]
        except (TypeError, ValueError):
            answer_str = str(correct_val)

        return {
            "question_type": "short_answer",
            "question_latex": question_text,
            "correct_answer": answer_str,
            "points": 0.5,
            "explanation": explanation,
            "difficulty": difficulty,
            "skill_id": skill_id,
            "skill_ids": [skill_id],
            "formula_ids": [],
            "sympy_expr": sympy_expr,
        }


def _perturb(expr):
    """Create a plausible but wrong variant of a SymPy expression."""
    strategies = [
        lambda e: e + random.choice([-2, -1, 1, 2]),
        lambda e: e * random.choice([-1, 2, Rational(1, 2)]),
        lambda e: -e,
        lambda e: e + random.choice([x, -x, 1, -1]),
    ]
    fn = random.choice(strategies)
    try:
        result = simplify(fn(expr))
        return result
    except Exception:
        return expr + random.randint(1, 3)


# ── Derivative Generator ────────────────────────────────
class DerivativeQuizGenerator(BaseQuizGenerator):
    skill_id = "derivative_basic"

    def _templates(self, difficulty: int) -> list:
        if difficulty <= 1:
            return [self._poly_easy]
        elif difficulty == 2:
            return [self._poly_medium, self._trig]
        else:
            return [self._composite, self._quotient]

    def _poly_easy(self) -> dict:
        a = random.randint(1, 6)
        n = random.randint(2, 4)
        b = random.randint(-5, 5)
        f = a * x**n + b * x
        f_prime = diff(f, x)
        q = f"Tính đạo hàm của $f(x) = {latex(f)}$"
        expl = f"$f'(x) = {latex(f_prime)}$"
        return self._make_mcq(f_prime, "derivative_basic", q, expl, 1, str(f))

    def _poly_medium(self) -> dict:
        a = random.randint(1, 4)
        b = random.randint(-3, 3)
        c = random.randint(-5, 5)
        f = a * x**3 + b * x**2 + c * x + random.randint(-3, 3)
        f_prime = diff(f, x)
        q = f"Tính đạo hàm của $f(x) = {latex(f)}$"
        expl = f"$f'(x) = {latex(f_prime)}$"
        return self._make_mcq(f_prime, "derivative_rules", q, expl, 2, str(f))

    def _trig(self) -> dict:
        funcs = [
            (sin(x), cos(x)),
            (cos(x), -sin(x)),
            (sin(2*x), 2*cos(2*x)),
            (cos(3*x), -3*sin(3*x)),
        ]
        f, f_prime = random.choice(funcs)
        a = random.randint(1, 5)
        f = a * f
        f_prime = a * f_prime
        q = f"Tính đạo hàm của $f(x) = {latex(f)}$"
        expl = f"$f'(x) = {latex(simplify(f_prime))}$"
        return self._make_mcq(simplify(f_prime), "derivative_rules", q, expl, 2, str(f))

    def _composite(self) -> dict:
        inner = random.choice([2*x + 1, x**2 + 1, 3*x - 2])
        outer_choices = [
            (lambda u: u**3, lambda u, u_prime: 3*u**2 * u_prime),
            (lambda u: sqrt(u), lambda u, u_prime: u_prime / (2 * sqrt(u))),
        ]
        outer_fn, outer_deriv = random.choice(outer_choices)
        f = outer_fn(inner)
        f_prime = simplify(diff(f, x))
        q = f"Tính đạo hàm của $f(x) = {latex(f)}$"
        expl = f"Áp dụng đạo hàm hàm hợp:\n$f'(x) = {latex(f_prime)}$"
        return self._make_mcq(f_prime, "derivative_rules", q, expl, 3, str(f))

    def _quotient(self) -> dict:
        num = random.choice([x + 1, 2*x + 3, x**2])
        den = random.choice([x - 1, x + 2, x**2 + 1])
        f = num / den
        f_prime = simplify(diff(f, x))
        q = f"Tính đạo hàm của $f(x) = {latex(f)}$"
        expl = f"Áp dụng công thức đạo hàm thương:\n$f'(x) = {latex(f_prime)}$"
        return self._make_mcq(f_prime, "derivative_rules", q, expl, 3, str(f))


# ── Integral Generator ──────────────────────────────────
class IntegralQuizGenerator(BaseQuizGenerator):
    skill_id = "integral_definite"

    def _templates(self, difficulty: int) -> list:
        if difficulty <= 1:
            return [self._primitive_easy]
        elif difficulty == 2:
            return [self._definite_medium]
        else:
            return [self._definite_hard]

    def _primitive_easy(self) -> dict:
        a = random.randint(1, 5)
        n = random.randint(1, 4)
        f = a * x**n
        F = integrate(f, x)
        q = f"Tìm nguyên hàm của $f(x) = {latex(f)}$"
        expl = f"$\\int {latex(f)} \\, dx = {latex(F)} + C$"
        return self._make_mcq(F, "primitive_basic", q, expl, 1, str(f))

    def _definite_medium(self) -> dict:
        a_coeff = random.randint(1, 3)
        n = random.randint(2, 3)
        f = a_coeff * x**n
        lower = random.choice([0, 1])
        upper = random.randint(lower + 1, lower + 3)
        result = integrate(f, (x, lower, upper))
        q = f"Tính $\\int_{{{lower}}}^{{{upper}}} {latex(f)} \\, dx$"
        expl = f"$\\int_{{{lower}}}^{{{upper}}} {latex(f)} \\, dx = {latex(result)}$"
        return self._make_mcq(result, "integral_definite", q, expl, 2, str(f))

    def _definite_hard(self) -> dict:
        funcs = [
            (sin(x), 0, sp.pi),
            (cos(x), 0, sp.pi / 2),
            (exp(x), 0, 1),
        ]
        f, lower, upper = random.choice(funcs)
        result = integrate(f, (x, lower, upper))
        q = f"Tính $\\int_{{{latex(lower)}}}^{{{latex(upper)}}} {latex(f)} \\, dx$"
        expl = f"$= {latex(result)}$"
        return self._make_mcq(result, "integral_definite", q, expl, 3, str(f))


# ── Equation Generator (Exponential/Log) ────────────────
class ExpLogQuizGenerator(BaseQuizGenerator):
    skill_id = "exp_log_equations"

    def _templates(self, difficulty: int) -> list:
        if difficulty <= 1:
            return [self._exp_easy]
        elif difficulty == 2:
            return [self._log_medium]
        else:
            return [self._exp_log_hard]

    def _exp_easy(self) -> dict:
        base = random.choice([2, 3, 5])
        result_exp = random.randint(1, 4)
        rhs = base ** result_exp
        q = f"Giải phương trình $${base}^x = {rhs}$$"
        answer = sp.Integer(result_exp)
        expl = f"${base}^x = {rhs} = {base}^{{{result_exp}}}$ nên $x = {result_exp}$"
        return self._make_mcq(answer, "exponential_basic", q, expl, 1)

    def _log_medium(self) -> dict:
        base = random.choice([2, 3, 10])
        arg = base ** random.randint(1, 4)
        result_val = sp.log(arg, base)
        base_str = "" if base == 10 else f"_{{{base}}}"
        q = f"Tính $\\log{base_str} {arg}$"
        expl = f"$\\log{base_str} {arg} = {latex(result_val)}$"
        return self._make_mcq(result_val, "logarithm_basic", q, expl, 2)

    def _exp_log_hard(self) -> dict:
        a = random.choice([2, 3])
        k = random.randint(1, 3)
        # Solve a^(2x) - k*a^x - (k+1) = 0  →  let t=a^x
        t = symbols("t", positive=True)
        eq = t**2 - (k + 1) * t - (k + 2)  # Designed to have t = k+2 as solution
        sols = solve(eq, t)
        pos_sols = [s for s in sols if s.is_positive]
        if pos_sols:
            t_val = pos_sols[0]
            x_val = simplify(sp.log(t_val, a))
        else:
            t_val = sp.Integer(k + 2)
            x_val = simplify(sp.log(t_val, a))
        q = (
            f"Giải phương trình "
            f"${latex(a)}^{{2x}} - {k+1} \\cdot {latex(a)}^x - {k+2} = 0$"
        )
        expl = f"Đặt $t = {a}^x > 0$, ta có $t^2 - {k+1}t - {k+2} = 0$.\nNghiệm dương $t = {latex(t_val)}$, suy ra $x = {latex(x_val)}$."
        return self._make_mcq(x_val, "exp_log_equations", q, expl, 3)


# ── Complex Number Generator ────────────────────────────
class ComplexQuizGenerator(BaseQuizGenerator):
    skill_id = "complex_basic"

    def _templates(self, difficulty: int) -> list:
        if difficulty <= 1:
            return [self._modulus]
        elif difficulty == 2:
            return [self._operations]
        else:
            return [self._equation]

    def _modulus(self) -> dict:
        a = random.randint(-5, 5)
        b = random.randint(1, 5) * random.choice([-1, 1])
        z = a + b * sp.I
        mod = simplify(sp.Abs(z))
        q = f"Tính mô-đun của số phức $z = {latex(z)}$"
        expl = f"$|z| = \\sqrt{{{a}^2 + {b}^2}} = {latex(mod)}$"
        return self._make_mcq(mod, "complex_basic", q, expl, 1, str(z))

    def _operations(self) -> dict:
        a1, b1 = random.randint(1, 4), random.randint(1, 4)
        a2, b2 = random.randint(1, 4), random.randint(1, 4)
        z1 = a1 + b1 * sp.I
        z2 = a2 + b2 * sp.I
        op = random.choice(["*", "+"])
        if op == "*":
            result = sp.expand(z1 * z2)
            q = f"Tính $z_1 \\cdot z_2$ biết $z_1 = {latex(z1)}$, $z_2 = {latex(z2)}$"
        else:
            result = z1 + z2
            q = f"Tính $z_1 + z_2$ biết $z_1 = {latex(z1)}$, $z_2 = {latex(z2)}$"
        expl = f"Kết quả: ${latex(result)}$"
        return self._make_mcq(result, "complex_operations", q, expl, 2)

    def _equation(self) -> dict:
        # z^2 + bz + c = 0 with complex roots
        z = symbols("z")
        r1 = random.randint(1, 3) + random.randint(1, 3) * sp.I
        r2 = sp.conjugate(r1)
        poly = sp.expand((z - r1) * (z - r2))
        coeffs = sp.Poly(poly, z).all_coeffs()
        q = f"Giải phương trình $z^2 + {latex(coeffs[1])}z + {latex(coeffs[2])} = 0$ trên $\\mathbb{{C}}$"
        expl = f"Nghiệm: $z = {latex(r1)}$ hoặc $z = {latex(r2)}$"
        return self._make_mcq(r1, "complex_equations", q, expl, 3, str(poly))


# ── Probability Generator ───────────────────────────────
class ProbabilityQuizGenerator(BaseQuizGenerator):
    skill_id = "probability_basic"

    def _templates(self, difficulty: int) -> list:
        if difficulty <= 1:
            return [self._basic_prob]
        else:
            return [self._distribution]

    def _basic_prob(self) -> dict:
        total = random.choice([6, 10, 12, 20, 36, 52])
        favorable = random.randint(1, total - 1)
        prob = Rational(favorable, total)
        scenarios = {
            6: f"Gieo xúc xắc. Tính xác suất để mặt xuất hiện chia hết cho số nào đó",
            36: f"Gieo hai xúc xắc cùng lúc",
            52: f"Rút ngẫu nhiên 1 lá bài từ bộ 52 lá",
        }
        context = scenarios.get(total, f"Trong {total} phần tử, chọn ngẫu nhiên 1 phần tử")
        q = f"{context}. Biết có {favorable} kết quả thuận lợi trong tổng {total} kết quả. Tính xác suất."
        expl = f"$P = \\frac{{{favorable}}}{{{total}}} = {latex(prob)}$"
        return self._make_mcq(prob, "probability_basic", q, expl, 1)

    def _distribution(self) -> dict:
        n = random.randint(3, 6)
        values = sorted(random.sample(range(1, 15), n))
        total = sum(values)
        mean = Rational(total, n)
        q = f"Tính trung bình cộng của mẫu số liệu: ${', '.join(str(v) for v in values)}$"
        expl = f"$\\bar{{x}} = \\frac{{{'+'.join(str(v) for v in values)}}}{{{n}}} = {latex(mean)}$"
        return self._make_mcq(mean, "statistics_descriptive", q, expl, 2)


# ── Sequence (Dãy số) Generator ─────────────────────────
class SequenceQuizGenerator(BaseQuizGenerator):
    """Generates questions about arithmetic and geometric sequences."""
    skill_id = "sequence_basic"

    def _templates(self, difficulty: int) -> list:
        if difficulty <= 1:
            return [self._arithmetic_nth_term]
        elif difficulty == 2:
            return [self._arithmetic_sum, self._geometric_nth_term]
        else:
            return [self._geometric_sum, self._mixed]

    def _arithmetic_nth_term(self) -> dict:
        u1 = random.randint(-5, 10)
        d = random.randint(-4, 6)
        while d == 0:
            d = random.randint(-4, 6)
        n = random.randint(5, 15)
        un = u1 + (n - 1) * d
        q = f"Cho CSC $(u_n)$ có $u_1 = {u1}$, công sai $d = {d}$. Tìm $u_{{{n}}}$."
        expl = f"$u_{{{n}}} = u_1 + ({n}-1) \\cdot d = {u1} + {n-1} \\cdot {d} = {un}$"
        return self._make_mcq(sp.Integer(un), "arithmetic_sequence", q, expl, 1)

    def _arithmetic_sum(self) -> dict:
        u1 = random.randint(1, 8)
        d = random.randint(1, 5)
        n = random.randint(5, 12)
        s_n = sp.Rational(n, 1) * (2 * u1 + (n - 1) * d) / 2
        q = f"Cho CSC $(u_n)$ có $u_1 = {u1}$, công sai $d = {d}$. Tính $S_{{{n}}}$."
        expl = f"$S_{{{n}}} = \\frac{{{n}}}{2}(2 \\cdot {u1} + ({n}-1) \\cdot {d}) = {latex(s_n)}$"
        return self._make_mcq(s_n, "arithmetic_sequence", q, expl, 2)

    def _geometric_nth_term(self) -> dict:
        u1 = random.choice([1, 2, 3, -1, -2])
        q_ratio = random.choice([2, 3, -2, sp.Rational(1, 2), sp.Rational(1, 3)])
        n = random.randint(3, 7)
        un = u1 * q_ratio**(n - 1)
        q_text = f"Cho CSN $(u_n)$ có $u_1 = {u1}$, công bội $q = {latex(q_ratio)}$. Tìm $u_{{{n}}}$."
        expl = f"$u_{{{n}}} = u_1 \\cdot q^{{{n}-1}} = {u1} \\cdot ({latex(q_ratio)})^{{{n-1}}} = {latex(simplify(un))}$"
        return self._make_mcq(simplify(un), "geometric_sequence", q_text, expl, 2)

    def _geometric_sum(self) -> dict:
        u1 = random.choice([1, 2, 3])
        q_ratio = random.choice([2, 3, sp.Rational(1, 2)])
        n = random.randint(3, 6)
        s_n = u1 * (1 - q_ratio**n) / (1 - q_ratio)
        q_text = f"Cho CSN $(u_n)$ có $u_1 = {u1}$, công bội $q = {latex(q_ratio)}$. Tính $S_{{{n}}}$."
        expl = f"$S_{{{n}}} = u_1 \\cdot \\frac{{1 - q^{{{n}}}}}{{1 - q}} = {latex(simplify(s_n))}$"
        return self._make_mcq(simplify(s_n), "geometric_sequence", q_text, expl, 3)

    def _mixed(self) -> dict:
        u1 = random.randint(1, 5)
        d = random.randint(1, 4)
        n_val = random.randint(8, 15)
        # Find n such that u_n > threshold
        threshold = random.randint(30, 60)
        n_solution = sp.ceiling((threshold - u1) / d) + 1
        q = f"Cho CSC $(u_n)$ có $u_1 = {u1}$, $d = {d}$. Tìm số hạng đầu tiên $u_n > {threshold}$."
        expl = f"$u_n = {u1} + (n-1) \\cdot {d} > {threshold}$ ⟹ $n > {latex((threshold - u1)/d + 1)}$ ⟹ $n = {latex(n_solution)}$"
        return self._make_mcq(n_solution, "sequence_basic", q, expl, 3)


# ── Geometry (Hình học không gian Oxyz) Generator ───────
class GeometryQuizGenerator(BaseQuizGenerator):
    """Generates questions about vectors, planes, lines in 3D space."""
    skill_id = "geometry_vectors"

    def _templates(self, difficulty: int) -> list:
        if difficulty <= 1:
            return [self._vector_dot_product, self._vector_length]
        elif difficulty == 2:
            return [self._plane_equation, self._line_parametric]
        else:
            return [self._distance_point_plane, self._sphere_equation]

    def _vector_dot_product(self) -> dict:
        a1, a2, a3 = [random.randint(-5, 5) for _ in range(3)]
        b1, b2, b3 = [random.randint(-5, 5) for _ in range(3)]
        dot = a1*b1 + a2*b2 + a3*b3
        q = f"Tính tích vô hướng $\\vec{{a}} \\cdot \\vec{{b}}$ biết $\\vec{{a}} = ({a1}; {a2}; {a3})$, $\\vec{{b}} = ({b1}; {b2}; {b3})$."
        expl = f"$\\vec{{a}} \\cdot \\vec{{b}} = {a1} \\cdot {b1} + {a2} \\cdot {b2} + {a3} \\cdot {b3} = {dot}$"
        return self._make_mcq(sp.Integer(dot), "geometry_vectors", q, expl, 1)

    def _vector_length(self) -> dict:
        a1, a2, a3 = [random.randint(-5, 5) for _ in range(3)]
        while a1 == 0 and a2 == 0 and a3 == 0:
            a1, a2, a3 = [random.randint(-5, 5) for _ in range(3)]
        length = sp.sqrt(a1**2 + a2**2 + a3**2)
        q = f"Tính $|\\vec{{a}}|$ biết $\\vec{{a}} = ({a1}; {a2}; {a3})$."
        expl = f"$|\\vec{{a}}| = \\sqrt{{{a1}^2 + {a2}^2 + {a3}^2}} = {latex(length)}$"
        return self._make_mcq(length, "geometry_vectors", q, expl, 1)

    def _plane_equation(self) -> dict:
        # Plane through point with normal vector
        x0, y0, z0 = [random.randint(-3, 3) for _ in range(3)]
        a, b, c = [random.randint(-4, 4) for _ in range(3)]
        while a == 0 and b == 0 and c == 0:
            a, b, c = [random.randint(-4, 4) for _ in range(3)]
        d = -(a * x0 + b * y0 + c * z0)
        q = (f"Viết phương trình mặt phẳng $(\\alpha)$ qua $M({x0}; {y0}; {z0})$ "
             f"có vectơ pháp tuyến $\\vec{{n}} = ({a}; {b}; {c})$. Tìm hệ số tự do $d$.")
        expl = f"$(\\alpha): {a}(x-{x0}) + {b}(y-{y0}) + {c}(z-{z0}) = 0 \\Rightarrow d = {d}$"
        return self._make_mcq(sp.Integer(d), "geometry_line_plane", q, expl, 2)

    def _line_parametric(self) -> dict:
        x0, y0, z0 = [random.randint(-3, 3) for _ in range(3)]
        a, b, c = [random.randint(-3, 3) for _ in range(3)]
        while a == 0 and b == 0 and c == 0:
            a, b, c = [random.randint(-3, 3) for _ in range(3)]
        t_val = random.randint(1, 3)
        point_on_line = (x0 + a * t_val, y0 + b * t_val, z0 + c * t_val)
        q = (f"Đường thẳng $d$ qua $M({x0}; {y0}; {z0})$ có VTCP $\\vec{{u}} = ({a}; {b}; {c})$. "
             f"Tìm hoành độ điểm thuộc $d$ khi $t = {t_val}$.")
        expl = f"$x = {x0} + {a} \\cdot {t_val} = {point_on_line[0]}$"
        return self._make_mcq(sp.Integer(point_on_line[0]), "geometry_line_plane", q, expl, 2)

    def _distance_point_plane(self) -> dict:
        a, b, c = [random.randint(1, 5) for _ in range(3)]
        d_coeff = random.randint(-10, 10)
        x0, y0, z0 = [random.randint(-5, 5) for _ in range(3)]
        numerator = abs(a * x0 + b * y0 + c * z0 + d_coeff)
        denominator = sp.sqrt(a**2 + b**2 + c**2)
        distance = sp.Rational(numerator, 1) / denominator
        q = (f"Tính khoảng cách từ $M({x0}; {y0}; {z0})$ đến mặt phẳng "
             f"$(\\alpha): {a}x + {b}y + {c}z + {d_coeff} = 0$.")
        expl = (f"$d(M, \\alpha) = \\frac{{|{a} \\cdot {x0} + {b} \\cdot {y0} + {c} \\cdot {z0} + {d_coeff}|}}"
                f"{{\\sqrt{{{a}^2 + {b}^2 + {c}^2}}}} = {latex(simplify(distance))}$")
        return self._make_mcq(simplify(distance), "geometry_distance_angle", q, expl, 3)

    def _sphere_equation(self) -> dict:
        a, b, c = [random.randint(-3, 3) for _ in range(3)]
        r = random.randint(1, 5)
        # (x-a)^2 + (y-b)^2 + (z-c)^2 = r^2
        r_sq = r**2
        q = (f"Mặt cầu tâm $I({a}; {b}; {c})$ bán kính $R = {r}$. "
             f"Tìm $R^2$.")
        expl = f"$R^2 = {r_sq}$"
        return self._make_mcq(sp.Integer(r_sq), "geometry_sphere", q, expl, 3)


# ── Function Survey (Khảo sát hàm số) Generator ────────
class FunctionSurveyQuizGenerator(BaseQuizGenerator):
    """Generates questions about function analysis: extrema, monotonicity, tangent lines."""
    skill_id = "function_survey"

    def _templates(self, difficulty: int) -> list:
        if difficulty <= 1:
            return [self._find_extrema_cubic]
        elif difficulty == 2:
            return [self._max_min_interval, self._tangent_line]
        else:
            return [self._intersection_count, self._fractional_extrema]

    def _find_extrema_cubic(self) -> dict:
        a = random.choice([1, -1, 2, -2])
        b = random.randint(-3, 3)
        c = random.randint(-6, 6)
        d_val = random.randint(-5, 5)
        f = a * x**3 + b * x**2 + c * x + d_val
        f_prime = diff(f, x)
        crits = solve(f_prime, x)
        real_crits = [c_val for c_val in crits if c_val.is_real]
        n_crits = len(real_crits)
        q = f"Hàm số $y = {latex(f)}$ có bao nhiêu điểm cực trị?"
        expl = f"$y' = {latex(f_prime)}$. Giải $y' = 0 \\Rightarrow$ {n_crits} nghiệm thực."
        return self._make_mcq(sp.Integer(n_crits), "function_survey", q, expl, 1)

    def _max_min_interval(self) -> dict:
        a = random.choice([1, -1])
        b = random.randint(-2, 2)
        c = random.randint(-4, 4)
        f = a * x**3 + b * x**2 + c * x
        lower = random.choice([-2, -1, 0])
        upper = random.choice([1, 2, 3])
        # Evaluate at critical points and endpoints
        f_prime = diff(f, x)
        crits = solve(f_prime, x)
        # Use float conversion to avoid SymPy comparison errors
        real_crits = []
        for c_val in crits:
            try:
                fv = float(c_val)
                if lower <= fv <= upper:
                    real_crits.append(c_val)
            except (TypeError, ValueError):
                pass
        candidates = [sp.Integer(lower), sp.Integer(upper)] + real_crits
        values = [f.subs(x, pt) for pt in candidates]
        # Use float for comparison, keep SymPy expr for the answer
        float_vals = [float(v) for v in values]
        max_idx = float_vals.index(max(float_vals))
        max_val = simplify(values[max_idx])
        q = f"Tìm GTLN của $f(x) = {latex(f)}$ trên $[{lower}; {upper}]$."
        expl = f"Tính $f'(x) = {latex(f_prime)}$, xét giá trị tại cực trị và đầu mút. GTLN = ${latex(max_val)}$."
        return self._make_mcq(max_val, "derivative_applications", q, expl, 2)

    def _tangent_line(self) -> dict:
        a = random.randint(1, 3)
        b = random.randint(-3, 3)
        c = random.randint(-5, 5)
        f = a * x**2 + b * x + c
        x0 = random.choice([-2, -1, 0, 1, 2])
        y0 = f.subs(x, x0)
        slope = diff(f, x).subs(x, x0)
        q = f"Viết phương trình tiếp tuyến của $y = {latex(f)}$ tại $x_0 = {x0}$. Tìm hệ số góc."
        expl = f"$y' = {latex(diff(f, x))}$, $y'({x0}) = {latex(slope)}$"
        return self._make_mcq(slope, "derivative_applications", q, expl, 2)

    def _intersection_count(self) -> dict:
        a = random.choice([1, -1])
        b = random.randint(-3, 0)
        f = a * x**3 + b * x
        m = random.choice([-2, -1, 0, 1, 2])
        # Count intersections of f(x) = m
        solutions = solve(f - m, x)
        n_roots = len([s for s in solutions if s.is_real])
        q = f"Đường thẳng $y = {m}$ cắt đồ thị $y = {latex(f)}$ tại bao nhiêu điểm?"
        expl = f"Giải ${latex(f)} = {m}$ → {n_roots} nghiệm thực."
        return self._make_mcq(sp.Integer(n_roots), "function_survey", q, expl, 3)

    def _fractional_extrema(self) -> dict:
        a, b_c = random.randint(1, 3), random.randint(-3, 3)
        c, d_c = random.randint(1, 3), random.randint(-3, 3)
        while a * d_c - b_c * c == 0:  # avoid constant function
            d_c = random.randint(-3, 3)
        f = (a * x + b_c) / (c * x + d_c)
        f_prime = simplify(diff(f, x))
        # Fractional linear has no extrema (f' never = 0 if ad-bc ≠ 0)
        has_extrema = 0
        q = f"Hàm số $y = {latex(f)}$ có bao nhiêu điểm cực trị?"
        expl = f"$y' = {latex(f_prime)}$. $y' \\neq 0\\ \\forall x \\in D \\Rightarrow$ không có cực trị."
        return self._make_mcq(sp.Integer(has_extrema), "function_survey", q, expl, 3)


# ── Statistics (Thống kê) Generator ─────────────────────
class StatisticsQuizGenerator(BaseQuizGenerator):
    """Generates questions about descriptive statistics: mean, median, mode, variance, std dev."""
    skill_id = "statistics_descriptive"

    def _templates(self, difficulty: int) -> list:
        if difficulty <= 1:
            return [self._mean, self._mode]
        elif difficulty == 2:
            return [self._median, self._quartiles]
        else:
            return [self._variance, self._std_dev]

    def _mean(self) -> dict:
        n = random.randint(4, 8)
        data = sorted(random.sample(range(1, 25), n))
        total = sum(data)
        mean = Rational(total, n)
        data_str = "; ".join(str(d) for d in data)
        q = f"Tính trung bình cộng của mẫu số liệu: ${data_str}$"
        expl = f"$\\bar{{x}} = \\frac{{{'+'.join(str(d) for d in data)}}}{{{n}}} = {latex(mean)}$"
        return self._make_mcq(mean, "statistics_descriptive", q, expl, 1)

    def _mode(self) -> dict:
        base = random.sample(range(1, 15), 4)
        mode_val = random.choice(base)
        data = base + [mode_val, mode_val]  # mode appears 3 times
        random.shuffle(data)
        data_str = "; ".join(str(d) for d in data)
        q = f"Tìm Mốt (Mode) của mẫu số liệu: ${data_str}$"
        expl = f"Giá trị ${mode_val}$ xuất hiện nhiều nhất (3 lần) → Mode = ${mode_val}$"
        return self._make_mcq(sp.Integer(mode_val), "statistics_descriptive", q, expl, 1)

    def _median(self) -> dict:
        n = random.choice([5, 7, 9])  # odd for unique median
        data = sorted(random.sample(range(1, 30), n))
        median_val = data[n // 2]
        data_str = "; ".join(str(d) for d in data)
        q = f"Tìm trung vị (Median) của mẫu số liệu: ${data_str}$"
        expl = f"Sắp xếp tăng dần, phần tử giữa (vị trí {n//2 + 1}) = ${median_val}$"
        return self._make_mcq(sp.Integer(median_val), "statistics_descriptive", q, expl, 2)

    def _quartiles(self) -> dict:
        n = 8
        data = sorted(random.sample(range(1, 30), n))
        # Q1 = median of lower half, Q3 = median of upper half
        lower = data[:n//2]
        upper = data[n//2:]
        q1 = Rational(lower[len(lower)//2 - 1] + lower[len(lower)//2], 2)
        q3 = Rational(upper[len(upper)//2 - 1] + upper[len(upper)//2], 2)
        iqr = q3 - q1
        data_str = "; ".join(str(d) for d in data)
        q = f"Cho mẫu số liệu: ${data_str}$. Tính khoảng tứ phân vị (IQR = $Q_3 - Q_1$)."
        expl = f"$Q_1 = {latex(q1)}$, $Q_3 = {latex(q3)}$ → IQR = ${latex(iqr)}$"
        return self._make_mcq(iqr, "statistics_descriptive", q, expl, 2)

    def _variance(self) -> dict:
        n = random.randint(4, 6)
        data = random.sample(range(1, 20), n)
        mean = Rational(sum(data), n)
        var = sum([(d - mean)**2 for d in data]) / n
        data_str = "; ".join(str(d) for d in data)
        q = f"Tính phương sai ($S^2$) của mẫu: ${data_str}$"
        expl = f"$\\bar{{x}} = {latex(mean)}$, $S^2 = \\frac{{1}}{{{n}}} \\sum(x_i - \\bar{{x}})^2 = {latex(simplify(var))}$"
        return self._make_mcq(simplify(var), "statistics_inference", q, expl, 3)

    def _std_dev(self) -> dict:
        n = random.randint(4, 5)
        # Use small integers for clean answers
        data = [random.randint(1, 10) for _ in range(n)]
        mean = Rational(sum(data), n)
        var = sum([(d - mean)**2 for d in data]) / n
        std = sp.sqrt(var)
        data_str = "; ".join(str(d) for d in data)
        q = f"Tính độ lệch chuẩn ($S$) của mẫu: ${data_str}$"
        expl = f"$S = \\sqrt{{S^2}} = \\sqrt{{{latex(simplify(var))}}} = {latex(simplify(std))}$"
        return self._make_mcq(simplify(std), "statistics_inference", q, expl, 3)


# ── Generator Registry ──────────────────────────────────
GENERATORS: dict[str, BaseQuizGenerator] = {
    # Đạo hàm
    "derivative_basic": DerivativeQuizGenerator(),
    "derivative_rules": DerivativeQuizGenerator(),
    "derivative_applications": FunctionSurveyQuizGenerator(),
    # Khảo sát hàm số
    "function_survey": FunctionSurveyQuizGenerator(),
    # Nguyên hàm & Tích phân
    "primitive_basic": IntegralQuizGenerator(),
    "integral_definite": IntegralQuizGenerator(),
    "integral_applications": IntegralQuizGenerator(),
    # Mũ & Logarit
    "exponential_basic": ExpLogQuizGenerator(),
    "logarithm_basic": ExpLogQuizGenerator(),
    "exp_log_equations": ExpLogQuizGenerator(),
    # Số phức
    "complex_basic": ComplexQuizGenerator(),
    "complex_operations": ComplexQuizGenerator(),
    "complex_equations": ComplexQuizGenerator(),
    # Xác suất & Thống kê
    "probability_basic": ProbabilityQuizGenerator(),
    "probability_distribution": ProbabilityQuizGenerator(),
    "statistics_descriptive": StatisticsQuizGenerator(),
    "statistics_inference": StatisticsQuizGenerator(),
    # Dãy số
    "sequence_basic": SequenceQuizGenerator(),
    "arithmetic_sequence": SequenceQuizGenerator(),
    "geometric_sequence": SequenceQuizGenerator(),
    # Hình học không gian
    "geometry_vectors": GeometryQuizGenerator(),
    "geometry_line_plane": GeometryQuizGenerator(),
    "geometry_distance_angle": GeometryQuizGenerator(),
    "geometry_sphere": GeometryQuizGenerator(),
}


def generate_questions(skill_id: str, difficulty: int = 1, count: int = 5,
                       question_type: str = "mcq") -> list[dict]:
    """
    Generate `count` quiz questions for a given skill, difficulty, and type.
    question_type: 'mcq' | 'true_false' | 'short_answer'
    """
    import traceback
    gen = GENERATORS.get(skill_id)
    if gen is None:
        print(f"[GENERATOR] WARNING: No generator for skill '{skill_id}', falling back to Derivative")
        gen = DerivativeQuizGenerator()

    questions = []
    for _ in range(count):
        try:
            if question_type == "true_false":
                q = _generate_true_false(gen, skill_id, difficulty)
            elif question_type == "short_answer":
                q = _generate_short_answer(gen, skill_id, difficulty)
            else:
                q = gen.generate(difficulty)
                q["question_type"] = "mcq"
            q["skill_id"] = skill_id
            q["skill_ids"] = q.get("skill_ids") or [skill_id]
            q["formula_ids"] = q.get("formula_ids") or []
            questions.append(q)
        except Exception as e:
            print(f"[GENERATOR] ERROR skill='{skill_id}' diff={difficulty}: {e}")
            traceback.print_exc()
            continue

    if not questions:
        print(f"[GENERATOR] FAILED 0 questions for skill='{skill_id}' diff={difficulty}")
    return questions



def generate_exam_questions(skill_id: str, difficulty: int = 1) -> list[dict]:
    """
    Generate a mixed set of questions in THPT QG exam format:
    - Part 1: MCQ (0.25 pts each)
    - Part 2: True/False with 4 statements (up to 1.0 pt each)
    - Part 3: Short answer (0.5 pts each)
    """
    mcq_qs = generate_questions(skill_id, difficulty, count=3, question_type="mcq")
    tf_qs = generate_questions(skill_id, difficulty, count=1, question_type="true_false")
    sa_qs = generate_questions(skill_id, difficulty, count=1, question_type="short_answer")
    return mcq_qs + tf_qs + sa_qs


# ── True/False Generator Helper ─────────────────────────
def _generate_true_false(gen: BaseQuizGenerator, skill_id: str, difficulty: int) -> dict:
    """Generate a True/False question with 4 statements about a math concept."""
    # Strategy: generate a function and create 4 verifiable statements
    a = random.randint(1, 4)
    b = random.randint(-3, 3)
    c = random.randint(-5, 5)
    d_val = random.randint(-3, 3)
    f = a * x**3 + b * x**2 + c * x + d_val
    f_prime = diff(f, x)

    statements = []

    # Statement 1: derivative claim
    f_prime_correct = simplify(f_prime)
    s1_is_correct = random.choice([True, False])
    if s1_is_correct:
        s1_text = f"$f'(x) = {latex(f_prime_correct)}$"
    else:
        wrong_prime = f_prime_correct + random.choice([1, -1, x])
        s1_text = f"$f'(x) = {latex(simplify(wrong_prime))}$"
    statements.append((s1_text, s1_is_correct))

    # Statement 2: critical points count
    crits = solve(f_prime, x)
    real_crits = [c for c in crits if c.is_real]
    actual_count = len(real_crits)
    s2_is_correct = random.choice([True, False])
    claimed_count = actual_count if s2_is_correct else (actual_count + random.choice([1, -1]))
    claimed_count = max(0, claimed_count)
    s2_text = f"Hàm số có {claimed_count} điểm cực trị"
    statements.append((s2_text, s2_is_correct))

    # Statement 3: value of f at x=0
    f_at_0 = f.subs(x, 0)
    s3_is_correct = random.choice([True, False])
    if s3_is_correct:
        s3_text = f"$f(0) = {latex(f_at_0)}$"
    else:
        s3_text = f"$f(0) = {latex(f_at_0 + random.choice([1, -1, 2]))}$"
    statements.append((s3_text, s3_is_correct))

    # Statement 4: limit or monotonicity
    s4_is_correct = random.choice([True, False])
    if a > 0:
        if s4_is_correct:
            s4_text = "$\\lim_{x \\to +\\infty} f(x) = +\\infty$"
        else:
            s4_text = "$\\lim_{x \\to +\\infty} f(x) = -\\infty$"
    else:
        if s4_is_correct:
            s4_text = "$\\lim_{x \\to +\\infty} f(x) = -\\infty$"
        else:
            s4_text = "$\\lim_{x \\to +\\infty} f(x) = +\\infty$"
    statements.append((s4_text, s4_is_correct))

    random.shuffle(statements)

    q_text = f"Cho hàm số $f(x) = {latex(f)}$. Xét tính Đúng/Sai của các mệnh đề sau:"
    correct_bools = [s[1] for s in statements]
    expl = f"Đạo hàm: $f'(x) = {latex(f_prime_correct)}$. Cực trị: {actual_count} điểm. $f(0) = {latex(f_at_0)}$."

    return gen._make_true_false(
        statements_data=statements,
        skill_id=skill_id,
        question_text=q_text,
        explanation=expl,
        difficulty=difficulty,
        sympy_expr=str(f),
    )


# ── Short Answer Generator Helper ───────────────────────
def _generate_short_answer(gen: BaseQuizGenerator, skill_id: str, difficulty: int) -> dict:
    """Generate a short-answer question requiring a numeric result."""
    # Strategy varies by difficulty
    if difficulty <= 1:
        # Simple derivative evaluation
        a = random.randint(1, 5)
        n = random.randint(2, 4)
        b = random.randint(-5, 5)
        f = a * x**n + b * x
        f_prime = diff(f, x)
        x_val = random.choice([1, 2, -1, 0])
        result = f_prime.subs(x, x_val)
        q = f"Tính $f'({x_val})$ biết $f(x) = {latex(f)}$"
        expl = f"$f'(x) = {latex(f_prime)}$ → $f'({x_val}) = {latex(result)}$"
    elif difficulty == 2:
        # Definite integral
        a_coeff = random.randint(1, 3)
        n = random.randint(1, 3)
        f = a_coeff * x**n
        lower = random.choice([0, 1])
        upper = random.randint(lower + 1, lower + 2)
        result = integrate(f, (x, lower, upper))
        q = f"Tính $\\int_{{{lower}}}^{{{upper}}} {latex(f)} \\, dx$"
        expl = f"$\\int_{{{lower}}}^{{{upper}}} {latex(f)} \\, dx = {latex(result)}$"
    else:
        # Complex modulus or exp/log
        a_val = random.randint(1, 5)
        b_val = random.randint(1, 5)
        z = a_val + b_val * sp.I
        result = simplify(sp.Abs(z))
        q = f"Tính mô-đun $|z|$ biết $z = {latex(z)}$"
        expl = f"$|z| = \\sqrt{{{a_val}^2 + {b_val}^2}} = {latex(result)}$"

    return gen._make_short_answer(
        correct_val=result,
        skill_id=skill_id,
        question_text=q,
        explanation=expl,
        difficulty=difficulty,
        sympy_expr=str(result),
    )
