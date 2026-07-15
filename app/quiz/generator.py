"""Quiz Generator — creates math questions using SymPy for exact answers."""

import random
import sympy as sp
from sympy import (
    symbols, diff, integrate, simplify, latex, Rational,
    sin, cos, exp, sqrt,
    solve,
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
                  explanation: str, difficulty: int, sympy_expr: str = "",
                  answer_kind: str = "auto", distractors: list = None) -> dict:
        """Build an MCQ dict with 4 choices (1 correct + 3 distractors).

        answer_kind — ràng buộc "hình dạng" của một đáp án nhiễu hợp lý:
          - "count":       số nguyên không âm (số cực trị, số nghiệm, số giao điểm...)
          - "nonneg":      số không âm (khoảng cách, độ dài, mô-đun, phương sai...)
          - "probability": số trong khoảng (0, 1]
          - "integer":     số nguyên bất kỳ
          - "value":       một giá trị số bất kỳ (kể cả số phức)
          - "expression":  biểu thức chứa biến
          - "auto":        tự suy ra từ correct_val
        distractors — các đáp án sai do template tính sẵn từ LỖI SAI ĐIỂN HÌNH
        của học sinh (quên nhân số mũ, nhầm dấu, quên chia...). Được ưu tiên
        dùng trước khi rơi về nhiễu ngẫu nhiên.
        """
        kind = _infer_kind(correct_val) if answer_kind == "auto" else answer_kind
        correct_latex = f"${latex(correct_val)}$"

        chosen: list = []
        chosen_latex: set = set()

        def _try_add(cand) -> None:
            if len(chosen) >= 3 or cand is None:
                return
            try:
                cand = sp.sympify(cand)
                if not _is_valid_distractor(cand, kind, correct_val):
                    return
                cand_latex = f"${latex(cand)}$"
            except Exception:
                return
            if cand_latex == correct_latex or cand_latex in chosen_latex:
                return
            chosen.append(cand)
            chosen_latex.add(cand_latex)

        # 1) Nhiễu từ lỗi sai điển hình (chất lượng cao nhất)
        for cand in (distractors or []):
            _try_add(cand)

        # 2) Nhiễu ngẫu nhiên nhưng tôn trọng answer_kind
        attempts = 0
        while len(chosen) < 3 and attempts < 40:
            attempts += 1
            _try_add(_fallback_distractor(correct_val, kind))

        # 3) Phương án cuối: dịch giá trị từng bước, vẫn giữ đúng answer_kind
        step = 1
        while len(chosen) < 3 and step < 100:
            for cand in _last_resort_candidates(correct_val, step, kind):
                _try_add(cand)
            step += 1

        # 4) Bảo đảm luôn đủ 4 lựa chọn duy nhất
        filler = 0
        while len(chosen) < 3:
            cand = sp.Integer(filler)
            cand_latex = f"${latex(cand)}$"
            if cand_latex != correct_latex and cand_latex not in chosen_latex:
                chosen.append(cand)
                chosen_latex.add(cand_latex)
            filler += 1

        # Đáp án số được sắp tăng dần (như đề thi chuẩn); biểu thức thì xáo trộn
        choices_vals = [correct_val] + chosen
        try:
            choices_vals.sort(key=lambda v: float(v))
        except (TypeError, ValueError):
            random.shuffle(choices_vals)
        choices = [f"${latex(v)}$" for v in choices_vals]
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


def _infer_kind(correct_val) -> str:
    """Infer the answer kind from the correct value when not given explicitly."""
    try:
        v = sp.sympify(correct_val)
        if v.is_Integer:
            return "integer"
        if v.is_number:
            return "value"
    except Exception:
        pass
    return "expression"


def _values_equal(a, b) -> bool:
    try:
        return sp.simplify(a - b) == 0
    except Exception:
        return latex(a) == latex(b)


def _is_valid_distractor(d, kind: str, correct) -> bool:
    """A distractor must be mathematically plausible for the question type:
    e.g. a 'count' question can never have a negative or fractional choice."""
    try:
        if _values_equal(d, correct):
            return False
        if kind == "expression":
            return True
        if kind == "count":
            return bool(d.is_integer) and bool(d >= 0)
        if kind == "probability":
            return bool(d.is_number) and bool(d > 0) and bool(d <= 1)
        if kind == "nonneg":
            return bool(d.is_number) and bool(d.is_nonnegative)
        if kind == "integer":
            return bool(d.is_integer)
        # "value": bất kỳ giá trị số nào (kể cả số phức), nhưng không chứa biến
        return bool(d.is_number)
    except Exception:
        return False


def _fallback_distractor(correct, kind: str):
    """Kind-aware random perturbation, used when the template didn't supply
    enough mistake-based distractors."""
    try:
        if kind == "count":
            n = int(correct)
            hi = max(3, n + 2)
            pool = [v for v in range(0, hi + 1) if v != n]
            return sp.Integer(random.choice(pool))
        if kind == "probability":
            p = sp.Rational(correct)
            cands = [1 - p,
                     sp.Rational(p.p, p.q + 1),
                     sp.Rational(p.p + 1, p.q),
                     sp.Rational(max(p.p - 1, 1), p.q),
                     p / 2]
            return random.choice(cands)
        if kind == "nonneg":
            return sp.Abs(_perturb_number(correct))
        if kind in ("integer", "value"):
            return _perturb_number(correct)
        return _perturb(correct)
    except Exception:
        return None


def _last_resort_candidates(correct, step: int, kind: str) -> list:
    """Deterministic shifts guaranteeing enough distinct, kind-valid values."""
    try:
        if kind == "count":
            return [sp.Integer(step - 1)]
        if kind == "probability":
            p = sp.Rational(correct)
            return [sp.Rational(p.p, p.q + step), sp.Rational(1, step + 1)]
        if kind == "nonneg":
            return [sp.Abs(correct + step), sp.Abs(correct - step)]
        return [correct + step, correct - step]
    except Exception:
        return []


def _perturb_number(c):
    """Small arithmetic slips on a numeric answer."""
    strategies = [
        lambda v: v + random.choice([-3, -2, -1, 1, 2, 3]),
        lambda v: -v,
        lambda v: 2 * v,
    ]
    return simplify(random.choice(strategies)(c))


def _perturb_coefficient(expr):
    """Flip the sign or scale a single term — mimics a per-term slip."""
    terms = sp.Add.make_args(sp.expand(expr))
    target = random.choice(terms)
    return expr - target + target * random.choice([-1, 2, 3])


def _perturb(expr):
    """Create a plausible but wrong variant of a SymPy expression,
    keeping the same overall 'shape' as the correct answer."""
    strategies = [
        lambda e: e + random.choice([-2, -1, 1, 2]),
        lambda e: e * random.choice([-1, 2]),
        lambda e: -e,
        _perturb_coefficient,
    ]
    fn = random.choice(strategies)
    try:
        return simplify(fn(expr))
    except Exception:
        try:
            return expr + random.randint(1, 3)
        except Exception:
            return None


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
        wrongs = [
            a * x**(n - 1) + b,          # quên nhân số mũ
            a * n * x**(n - 1) + b * x,  # quên đạo hàm hạng tử bx
            a * n * x**(n - 1) - b,      # nhầm dấu hệ số b
        ]
        return self._make_mcq(f_prime, "derivative_basic", q, expl, 1, str(f),
                              answer_kind="expression", distractors=wrongs)

    def _poly_medium(self) -> dict:
        a = random.randint(1, 4)
        b = random.randint(-3, 3)
        c = random.randint(-5, 5)
        d_val = random.randint(-3, 3)
        f = a * x**3 + b * x**2 + c * x + d_val
        f_prime = diff(f, x)
        q = f"Tính đạo hàm của $f(x) = {latex(f)}$"
        expl = f"$f'(x) = {latex(f_prime)}$"
        wrongs = [
            f_prime + d_val,                  # quên bỏ hằng số
            a * x**2 + b * x + c,             # quên nhân số mũ
            3 * a * x**2 + b * x + c,         # quên nhân 2 ở hạng tử bx²
            3 * a * x**2 + 2 * b * x - c,     # nhầm dấu hệ số c
        ]
        return self._make_mcq(f_prime, "derivative_rules", q, expl, 2, str(f),
                              answer_kind="expression", distractors=wrongs)

    def _trig(self) -> dict:
        # (hàm, đạo hàm đúng, đạo hàm khi quên nhân hệ số trong — lỗi điển hình)
        funcs = [
            (sin(x), cos(x), -cos(x)),
            (cos(x), -sin(x), sin(x)),
            (sin(2*x), 2*cos(2*x), cos(2*x)),
            (cos(3*x), -3*sin(3*x), sin(3*x)),
        ]
        f0, f0_prime, f0_wrong = random.choice(funcs)
        a = random.randint(1, 5)
        f = a * f0
        f_prime = simplify(a * f0_prime)
        q = f"Tính đạo hàm của $f(x) = {latex(f)}$"
        expl = f"$f'(x) = {latex(f_prime)}$"
        wrongs = [-f_prime, a * f0_wrong, -a * f0_wrong, a * f0, -a * f0]
        return self._make_mcq(f_prime, "derivative_rules", q, expl, 2, str(f),
                              answer_kind="expression", distractors=wrongs)

    def _composite(self) -> dict:
        inner = random.choice([2*x + 1, x**2 + 1, 3*x - 2])
        outer_choices = [
            (lambda u: u**3, lambda u, u_prime: 3*u**2 * u_prime),
            (lambda u: sqrt(u), lambda u, u_prime: u_prime / (2 * sqrt(u))),
        ]
        outer_fn, outer_deriv = random.choice(outer_choices)
        f = outer_fn(inner)
        f_prime = simplify(diff(f, x))
        no_chain = simplify(outer_deriv(inner, 1))  # quên nhân u'
        q = f"Tính đạo hàm của $f(x) = {latex(f)}$"
        expl = f"Áp dụng đạo hàm hàm hợp:\n$f'(x) = {latex(f_prime)}$"
        wrongs = [no_chain, -f_prime, simplify(no_chain * inner)]  # nhân nhầm u thay vì u'
        return self._make_mcq(f_prime, "derivative_rules", q, expl, 3, str(f),
                              answer_kind="expression", distractors=wrongs)

    def _quotient(self) -> dict:
        num = random.choice([x + 1, 2*x + 3, x**2])
        den = random.choice([x - 1, x + 2, x**2 + 1])
        f = num / den
        f_prime = simplify(diff(f, x))
        q = f"Tính đạo hàm của $f(x) = {latex(f)}$"
        expl = f"Áp dụng công thức đạo hàm thương:\n$f'(x) = {latex(f_prime)}$"
        wrongs = [
            simplify(-f_prime),                                            # đảo tử số (v'u − u'v)
            simplify((diff(num, x) * den + num * diff(den, x)) / den**2),  # nhầm dấu +
            simplify(diff(num, x) / diff(den, x)),                         # đạo hàm tử / đạo hàm mẫu
        ]
        return self._make_mcq(f_prime, "derivative_rules", q, expl, 3, str(f),
                              answer_kind="expression", distractors=wrongs)


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
        wrongs = [
            a * x**(n + 1),                    # quên chia cho n+1
            a * n * x**(n - 1),                # đạo hàm thay vì nguyên hàm
            Rational(a, n + 1) * x**n,         # quên tăng số mũ
        ]
        return self._make_mcq(F, "primitive_basic", q, expl, 1, str(f),
                              answer_kind="expression", distractors=wrongs)

    def _definite_medium(self) -> dict:
        a_coeff = random.randint(1, 3)
        n = random.randint(2, 3)
        f = a_coeff * x**n
        lower = random.choice([0, 1])
        upper = random.randint(lower + 1, lower + 3)
        result = integrate(f, (x, lower, upper))
        F = integrate(f, x)
        Fb, Fa = F.subs(x, upper), F.subs(x, lower)
        q = f"Tính $\\int_{{{lower}}}^{{{upper}}} {latex(f)} \\, dx$"
        expl = f"$\\int_{{{lower}}}^{{{upper}}} {latex(f)} \\, dx = {latex(result)}$"
        wrongs = [Fb + Fa, Fa - Fb, Fb]  # cộng thay vì trừ, đảo cận, quên cận dưới
        return self._make_mcq(result, "integral_definite", q, expl, 2, str(f),
                              answer_kind="value", distractors=wrongs)

    def _definite_hard(self) -> dict:
        funcs = [
            (sin(x), 0, sp.pi),
            (cos(x), 0, sp.pi / 2),
            (exp(x), 0, 1),
        ]
        f, lower, upper = random.choice(funcs)
        result = integrate(f, (x, lower, upper))
        F = integrate(f, x)
        Fb, Fa = F.subs(x, upper), F.subs(x, lower)
        q = f"Tính $\\int_{{{latex(lower)}}}^{{{latex(upper)}}} {latex(f)} \\, dx$"
        expl = f"$= {latex(result)}$"
        wrongs = [simplify(Fb + Fa), simplify(Fa - Fb), simplify(Fb)]
        return self._make_mcq(result, "integral_definite", q, expl, 3, str(f),
                              answer_kind="value", distractors=wrongs)


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
        wrongs = [result_exp + 1, result_exp - 1, -result_exp]
        return self._make_mcq(answer, "exponential_basic", q, expl, 1,
                              answer_kind="integer", distractors=wrongs)

    def _log_medium(self) -> dict:
        base = random.choice([2, 3, 10])
        k = random.randint(1, 4)
        arg = base ** k
        result_val = sp.Integer(k)
        base_str = "" if base == 10 else f"_{{{base}}}"
        q = f"Tính $\\log{base_str} {arg}$"
        expl = f"$\\log{base_str} {arg} = \\log{base_str} {base}^{{{k}}} = {k}$"
        wrongs = [k + 1, k - 1, base, arg // base]
        return self._make_mcq(result_val, "logarithm_basic", q, expl, 2,
                              answer_kind="integer", distractors=wrongs)

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
        wrongs = [t_val, -x_val, x_val + 1]  # quên đổi về x, nhầm dấu, lệch 1
        return self._make_mcq(x_val, "exp_log_equations", q, expl, 3,
                              answer_kind="value", distractors=wrongs)


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
        wrongs = [
            sp.Integer(a**2 + b**2),               # quên lấy căn
            sp.Integer(abs(a) + abs(b)),           # cộng trị tuyệt đối
            sp.sqrt(abs(a**2 - b**2)),             # nhầm dấu trong căn
        ]
        return self._make_mcq(mod, "complex_basic", q, expl, 1, str(z),
                              answer_kind="nonneg", distractors=wrongs)

    def _operations(self) -> dict:
        a1, b1 = random.randint(1, 4), random.randint(1, 4)
        a2, b2 = random.randint(1, 4), random.randint(1, 4)
        z1 = a1 + b1 * sp.I
        z2 = a2 + b2 * sp.I
        op = random.choice(["*", "+"])
        if op == "*":
            result = sp.expand(z1 * z2)
            q = f"Tính $z_1 \\cdot z_2$ biết $z_1 = {latex(z1)}$, $z_2 = {latex(z2)}$"
            wrongs = [
                (a1*a2 + b1*b2) + (a1*b2 + a2*b1) * sp.I,  # quên i² = −1
                a1*a2 + b1*b2 * sp.I,                       # nhân từng thành phần
                z1 + z2,                                    # cộng thay vì nhân
            ]
        else:
            result = z1 + z2
            q = f"Tính $z_1 + z_2$ biết $z_1 = {latex(z1)}$, $z_2 = {latex(z2)}$"
            wrongs = [z1 - z2, z2 - z1, sp.conjugate(z1 + z2)]
        expl = f"Kết quả: ${latex(result)}$"
        return self._make_mcq(result, "complex_operations", q, expl, 2,
                              answer_kind="value", distractors=wrongs)

    def _equation(self) -> dict:
        # z^2 + bz + c = 0 with complex roots
        z = symbols("z")
        r1 = random.randint(1, 3) + random.randint(1, 3) * sp.I
        r2 = sp.conjugate(r1)
        poly = sp.expand((z - r1) * (z - r2))
        q = (f"Phương trình ${latex(poly)} = 0$ "
             f"có nghiệm phức với phần ảo dương là:")
        expl = f"Hai nghiệm: $z = {latex(r1)}$ và $z = {latex(r2)}$. Nghiệm có phần ảo dương: $z = {latex(r1)}$"
        re_p, im_p = r1.as_real_imag()
        wrongs = [
            r2,                       # chọn nhầm nghiệm liên hợp
            -re_p + im_p * sp.I,      # nhầm dấu phần thực
            im_p + re_p * sp.I,       # hoán đổi phần thực / phần ảo
        ]
        return self._make_mcq(r1, "complex_equations", q, expl, 3, str(poly),
                              answer_kind="value", distractors=wrongs)


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
            6: "Gieo xúc xắc. Tính xác suất để mặt xuất hiện chia hết cho số nào đó",
            36: "Gieo hai xúc xắc cùng lúc",
            52: "Rút ngẫu nhiên 1 lá bài từ bộ 52 lá",
        }
        context = scenarios.get(total, f"Trong {total} phần tử, chọn ngẫu nhiên 1 phần tử")
        q = f"{context}. Biết có {favorable} kết quả thuận lợi trong tổng {total} kết quả. Tính xác suất."
        expl = f"$P = \\frac{{{favorable}}}{{{total}}} = {latex(prob)}$"
        wrongs = [
            1 - prob,                                   # xác suất biến cố đối
            Rational(favorable, total - favorable),     # nhầm tỉ lệ thuận lợi/bất lợi
            Rational(favorable + 1, total),
            Rational(max(favorable - 1, 1), total),
        ]
        return self._make_mcq(prob, "probability_basic", q, expl, 1,
                              answer_kind="probability", distractors=wrongs)

    def _distribution(self) -> dict:
        n = random.randint(3, 6)
        values = sorted(random.sample(range(1, 15), n))
        total = sum(values)
        mean = Rational(total, n)
        q = f"Tính trung bình cộng của mẫu số liệu: ${', '.join(str(v) for v in values)}$"
        expl = f"$\\bar{{x}} = \\frac{{{'+'.join(str(v) for v in values)}}}{{{n}}} = {latex(mean)}$"
        wrongs = [
            Rational(total, n - 1),          # chia nhầm cho n−1
            Rational(total, n + 1),          # chia nhầm cho n+1
            sp.Integer(values[n // 2]),      # nhầm với trung vị
        ]
        return self._make_mcq(mean, "statistics_descriptive", q, expl, 2,
                              answer_kind="nonneg", distractors=wrongs)


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
        wrongs = [u1 + n * d, u1 + (n - 2) * d, u1 - (n - 1) * d]  # lệch 1 số hạng, nhầm dấu d
        return self._make_mcq(sp.Integer(un), "arithmetic_sequence", q, expl, 1,
                              answer_kind="integer", distractors=wrongs)

    def _arithmetic_sum(self) -> dict:
        u1 = random.randint(1, 8)
        d = random.randint(1, 5)
        n = random.randint(5, 12)
        s_n = sp.Rational(n, 1) * (2 * u1 + (n - 1) * d) / 2
        q = f"Cho CSC $(u_n)$ có $u_1 = {u1}$, công sai $d = {d}$. Tính $S_{{{n}}}$."
        expl = f"$S_{{{n}}} = \\frac{{{n}}}{2}(2 \\cdot {u1} + ({n}-1) \\cdot {d}) = {latex(s_n)}$"
        un = u1 + (n - 1) * d
        wrongs = [
            sp.Integer(n * (2 * u1 + (n - 1) * d)),  # quên chia 2
            sp.Integer(un),                           # nhầm với số hạng u_n
            s_n - un,                                 # tính nhầm S_{n−1}
        ]
        return self._make_mcq(s_n, "arithmetic_sequence", q, expl, 2,
                              answer_kind="value", distractors=wrongs)

    def _geometric_nth_term(self) -> dict:
        u1 = random.choice([1, 2, 3, -1, -2])
        q_ratio = random.choice([2, 3, -2, sp.Rational(1, 2), sp.Rational(1, 3)])
        n = random.randint(3, 7)
        un = u1 * q_ratio**(n - 1)
        q_text = f"Cho CSN $(u_n)$ có $u_1 = {u1}$, công bội $q = {latex(q_ratio)}$. Tìm $u_{{{n}}}$."
        expl = f"$u_{{{n}}} = u_1 \\cdot q^{{{n}-1}} = {u1} \\cdot ({latex(q_ratio)})^{{{n-1}}} = {latex(simplify(un))}$"
        wrongs = [
            simplify(u1 * q_ratio**n),        # nhầm mũ n
            simplify(u1 * q_ratio**(n - 2)),  # nhầm mũ n−2
            simplify(u1 * q_ratio * (n - 1)), # nhân thay vì lũy thừa (nhầm sang CSC)
        ]
        return self._make_mcq(simplify(un), "geometric_sequence", q_text, expl, 2,
                              answer_kind="value", distractors=wrongs)

    def _geometric_sum(self) -> dict:
        u1 = random.choice([1, 2, 3])
        q_ratio = random.choice([2, 3, sp.Rational(1, 2)])
        n = random.randint(3, 6)
        s_n = u1 * (1 - q_ratio**n) / (1 - q_ratio)
        q_text = f"Cho CSN $(u_n)$ có $u_1 = {u1}$, công bội $q = {latex(q_ratio)}$. Tính $S_{{{n}}}$."
        expl = f"$S_{{{n}}} = u_1 \\cdot \\frac{{1 - q^{{{n}}}}}{{1 - q}} = {latex(simplify(s_n))}$"
        wrongs = [
            simplify(u1 * (1 - q_ratio**(n - 1)) / (1 - q_ratio)),  # tính nhầm S_{n−1}
            simplify(u1 * q_ratio**(n - 1)),                        # nhầm với u_n
            simplify(u1 * q_ratio**n),
        ]
        return self._make_mcq(simplify(s_n), "geometric_sequence", q_text, expl, 3,
                              answer_kind="value", distractors=wrongs)

    def _mixed(self) -> dict:
        u1 = random.randint(1, 5)
        d = random.randint(1, 4)
        # Find n such that u_n > threshold
        threshold = random.randint(30, 60)
        n_solution = sp.ceiling((threshold - u1) / d) + 1
        q = f"Cho CSC $(u_n)$ có $u_1 = {u1}$, $d = {d}$. Tìm số hạng đầu tiên $u_n > {threshold}$."
        expl = f"$u_n = {u1} + (n-1) \\cdot {d} > {threshold}$ ⟹ $n > {latex((threshold - u1)/d + 1)}$ ⟹ $n = {latex(n_solution)}$"
        wrongs = [n_solution - 1, n_solution + 1, sp.ceiling((threshold - u1) / d)]  # quên +1
        return self._make_mcq(n_solution, "sequence_basic", q, expl, 3,
                              answer_kind="count", distractors=wrongs)


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
        wrongs = [
            a1*b1 + a2*b2,             # quên thành phần thứ ba
            a1*b1 + a2*b2 - a3*b3,     # nhầm dấu một thành phần
            -dot,
        ]
        return self._make_mcq(sp.Integer(dot), "geometry_vectors", q, expl, 1,
                              answer_kind="integer", distractors=wrongs)

    def _vector_length(self) -> dict:
        a1, a2, a3 = [random.randint(-5, 5) for _ in range(3)]
        while a1 == 0 and a2 == 0 and a3 == 0:
            a1, a2, a3 = [random.randint(-5, 5) for _ in range(3)]
        length = sp.sqrt(a1**2 + a2**2 + a3**2)
        q = f"Tính $|\\vec{{a}}|$ biết $\\vec{{a}} = ({a1}; {a2}; {a3})$."
        expl = f"$|\\vec{{a}}| = \\sqrt{{{a1}^2 + {a2}^2 + {a3}^2}} = {latex(length)}$"
        wrongs = [
            sp.Integer(a1**2 + a2**2 + a3**2),          # quên lấy căn
            sp.Integer(abs(a1) + abs(a2) + abs(a3)),    # cộng trị tuyệt đối
            sp.sqrt(abs(a1**2 + a2**2 - a3**2)),        # nhầm dấu trong căn
        ]
        return self._make_mcq(length, "geometry_vectors", q, expl, 1,
                              answer_kind="nonneg", distractors=wrongs)

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
        wrongs = [-d, d + a, d - a]  # nhầm dấu khi khai triển, cộng sót một hạng tử
        return self._make_mcq(sp.Integer(d), "geometry_line_plane", q, expl, 2,
                              answer_kind="integer", distractors=wrongs)

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
        wrongs = [
            point_on_line[1],      # nhầm sang tung độ
            point_on_line[2],      # nhầm sang cao độ
            x0 - a * t_val,        # nhầm dấu tham số
        ]
        return self._make_mcq(sp.Integer(point_on_line[0]), "geometry_line_plane", q, expl, 2,
                              answer_kind="integer", distractors=wrongs)

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
        wrongs = [
            sp.Rational(numerator, a**2 + b**2 + c**2),  # quên căn ở mẫu
            sp.Integer(numerator),                        # quên chia cho |n|
            simplify(2 * distance),
        ]
        return self._make_mcq(simplify(distance), "geometry_distance_angle", q, expl, 3,
                              answer_kind="nonneg", distractors=wrongs)

    def _sphere_equation(self) -> dict:
        a, b, c = [random.randint(-3, 3) for _ in range(3)]
        r = random.randint(1, 5)
        # (x-a)^2 + (y-b)^2 + (z-c)^2 = r^2
        r_sq = r**2
        q = (f"Mặt cầu tâm $I({a}; {b}; {c})$ bán kính $R = {r}$. "
             f"Tìm $R^2$.")
        expl = f"$R^2 = {r_sq}$"
        wrongs = [r, 2 * r, 4 * r**2]  # nhầm R, đường kính, (2R)²
        return self._make_mcq(sp.Integer(r_sq), "geometry_sphere", q, expl, 3,
                              answer_kind="nonneg", distractors=wrongs)


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
        # Hàm bậc 3 có cực trị ⟺ y' = 3ax² + 2bx + c có 2 nghiệm phân biệt (Δ' = b² − 3ac > 0);
        # nghiệm kép của y' KHÔNG phải cực trị vì y' không đổi dấu.
        disc = b**2 - 3 * a * c
        n_extrema = 2 if disc > 0 else 0
        q = f"Hàm số $y = {latex(f)}$ có bao nhiêu điểm cực trị?"
        if n_extrema == 2:
            expl = (f"$y' = {latex(f_prime)}$ có $\\Delta' = b^2 - 3ac = {disc} > 0$ nên $y'$ có "
                    f"2 nghiệm phân biệt và đổi dấu qua mỗi nghiệm $\\Rightarrow$ 2 điểm cực trị.")
        else:
            expl = (f"$y' = {latex(f_prime)}$ có $\\Delta' = b^2 - 3ac = {disc} \\le 0$ nên $y'$ "
                    f"không đổi dấu $\\Rightarrow$ hàm số không có cực trị.")
        wrongs = [v for v in (0, 1, 2, 3) if v != n_extrema]
        return self._make_mcq(sp.Integer(n_extrema), "function_survey", q, expl, 1,
                              str(f), answer_kind="count", distractors=wrongs)

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
        min_val = simplify(values[float_vals.index(min(float_vals))])
        q = f"Tìm GTLN của $f(x) = {latex(f)}$ trên $[{lower}; {upper}]$."
        expl = f"Tính $f'(x) = {latex(f_prime)}$, xét giá trị tại cực trị và đầu mút. GTLN = ${latex(max_val)}$."
        # Nhiễu: GTNN (nhầm max/min) và các giá trị thật tại đầu mút / điểm dừng
        wrongs = [min_val] + [simplify(v) for v in values]
        return self._make_mcq(max_val, "derivative_applications", q, expl, 2,
                              str(f), answer_kind="value", distractors=wrongs)

    def _tangent_line(self) -> dict:
        a = random.randint(1, 3)
        b = random.randint(-3, 3)
        c = random.randint(-5, 5)
        f = a * x**2 + b * x + c
        x0 = random.choice([-2, -1, 0, 1, 2])
        slope = diff(f, x).subs(x, x0)
        q = f"Viết phương trình tiếp tuyến của $y = {latex(f)}$ tại $x_0 = {x0}$. Tìm hệ số góc."
        expl = f"$y' = {latex(diff(f, x))}$, $y'({x0}) = {latex(slope)}$"
        wrongs = [
            f.subs(x, x0),        # nhầm f(x₀) với f'(x₀)
            -slope,               # nhầm dấu
            2 * a * x0 - b,       # nhầm dấu hệ số b khi thế
        ]
        return self._make_mcq(slope, "derivative_applications", q, expl, 2,
                              str(f), answer_kind="integer", distractors=wrongs)

    def _intersection_count(self) -> dict:
        a = random.choice([1, -1])
        b = random.randint(-3, 0)
        f = a * x**3 + b * x
        m = random.choice([-2, -1, 0, 1, 2])
        # Đếm nghiệm thực PHÂN BIỆT (real_roots chính xác cả khi nghiệm là căn thức phức tạp)
        n_roots = len(set(sp.Poly(f - m, x).real_roots()))
        q = f"Đường thẳng $y = {m}$ cắt đồ thị $y = {latex(f)}$ tại bao nhiêu điểm?"
        expl = f"Giải ${latex(f)} = {m}$ → {n_roots} nghiệm thực phân biệt."
        wrongs = [v for v in (0, 1, 2, 3) if v != n_roots]
        return self._make_mcq(sp.Integer(n_roots), "function_survey", q, expl, 3,
                              str(f), answer_kind="count", distractors=wrongs)

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
        return self._make_mcq(sp.Integer(has_extrema), "function_survey", q, expl, 3,
                              str(f), answer_kind="count", distractors=[1, 2, 3])


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
        wrongs = [
            Rational(total, n - 1),        # chia nhầm cho n−1
            Rational(total, n + 1),        # chia nhầm cho n+1
            sp.Integer(data[n // 2]),      # nhầm với trung vị
        ]
        return self._make_mcq(mean, "statistics_descriptive", q, expl, 1,
                              answer_kind="nonneg", distractors=wrongs)

    def _mode(self) -> dict:
        base = random.sample(range(1, 15), 4)
        mode_val = random.choice(base)
        data = base + [mode_val, mode_val]  # mode appears 3 times
        random.shuffle(data)
        data_str = "; ".join(str(d) for d in data)
        q = f"Tìm Mốt (Mode) của mẫu số liệu: ${data_str}$"
        expl = f"Giá trị ${mode_val}$ xuất hiện nhiều nhất (3 lần) → Mode = ${mode_val}$"
        # Nhiễu: các giá trị khác CÓ MẶT trong mẫu — buộc học sinh đếm tần số thật
        wrongs = [v for v in base if v != mode_val]
        return self._make_mcq(sp.Integer(mode_val), "statistics_descriptive", q, expl, 1,
                              answer_kind="integer", distractors=wrongs)

    def _median(self) -> dict:
        n = random.choice([5, 7, 9])  # odd for unique median
        data = sorted(random.sample(range(1, 30), n))
        median_val = data[n // 2]
        data_str = "; ".join(str(d) for d in data)
        q = f"Tìm trung vị (Median) của mẫu số liệu: ${data_str}$"
        expl = f"Sắp xếp tăng dần, phần tử giữa (vị trí {n//2 + 1}) = ${median_val}$"
        wrongs = [
            Rational(sum(data), n),    # nhầm với trung bình cộng
            data[n // 2 - 1],          # phần tử liền trước
            data[n // 2 + 1],          # phần tử liền sau
        ]
        return self._make_mcq(sp.Integer(median_val), "statistics_descriptive", q, expl, 2,
                              answer_kind="value", distractors=wrongs)

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
        wrongs = [
            q1,                              # trả lời nhầm Q1
            q3,                              # trả lời nhầm Q3
            sp.Integer(data[-1] - data[0]),  # nhầm với khoảng biến thiên
        ]
        return self._make_mcq(iqr, "statistics_descriptive", q, expl, 2,
                              answer_kind="nonneg", distractors=wrongs)

    def _variance(self) -> dict:
        n = random.randint(4, 6)
        data = random.sample(range(1, 20), n)
        mean = Rational(sum(data), n)
        var = sum([(d - mean)**2 for d in data]) / n
        data_str = "; ".join(str(d) for d in data)
        q = f"Tính phương sai ($S^2$) của mẫu: ${data_str}$"
        expl = f"$\\bar{{x}} = {latex(mean)}$, $S^2 = \\frac{{1}}{{{n}}} \\sum(x_i - \\bar{{x}})^2 = {latex(simplify(var))}$"
        wrongs = [
            simplify(sp.sqrt(var)),                                   # nhầm với độ lệch chuẩn
            simplify(sum((d - mean)**2 for d in data) / (n - 1)),     # chia nhầm cho n−1
            mean,                                                     # nhầm với trung bình
        ]
        return self._make_mcq(simplify(var), "statistics_inference", q, expl, 3,
                              answer_kind="nonneg", distractors=wrongs)

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
        wrongs = [
            simplify(var),                                                    # quên lấy căn
            simplify(sp.sqrt(sum((d - mean)**2 for d in data) / (n - 1))),    # chia nhầm cho n−1
            mean,                                                             # nhầm với trung bình
        ]
        return self._make_mcq(simplify(std), "statistics_inference", q, expl, 3,
                              answer_kind="nonneg", distractors=wrongs)


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

    # Statement 2: extrema count — cực trị ⟺ y' có 2 nghiệm phân biệt (Δ' = b² − 3ac > 0)
    disc = b**2 - 3 * a * c
    actual_count = 2 if disc > 0 else 0
    s2_is_correct = random.choice([True, False])
    if s2_is_correct:
        claimed_count = actual_count
    else:
        # Số "sai" phải KHÁC số thật — tránh sinh mệnh đề đúng nhưng bị chấm sai
        claimed_count = random.choice([v for v in (0, 1, 2, 3) if v != actual_count])
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
