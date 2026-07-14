"""Shared deterministic scoring rules for practice quizzes and exams."""

from __future__ import annotations


TF_SCORE_TABLE = {
    0: 0.0,
    1: 0.1,
    2: 0.25,
    3: 0.5,
    4: 1.0,
}


def score_true_false(
    student_answers: list[bool],
    correct_answers: list[bool],
) -> tuple[float, int]:
    """Return THPT points and the number of correctly judged statements."""
    num_correct = sum(
        student == correct
        for student, correct in zip(student_answers, correct_answers)
    )
    return TF_SCORE_TABLE.get(num_correct, 0.0), num_correct


def answers_match(
    student_answer: str,
    correct_answer: str,
    *,
    numeric_tolerance: float = 0.01,
    remove_spaces: bool = False,
) -> bool:
    """Compare numeric answers first, then normalized text answers."""
    student = str(student_answer).strip().replace(",", ".")
    correct = str(correct_answer).strip().replace(",", ".")
    if remove_spaces:
        student = student.replace(" ", "")
        correct = correct.replace(" ", "")

    try:
        student_value = float(student)
        correct_value = float(correct)
        if numeric_tolerance > 0:
            return abs(student_value - correct_value) < numeric_tolerance
        return student_value == correct_value
    except (TypeError, ValueError):
        return student.casefold() == correct.casefold()


def score_short_answer(
    student_answer: str,
    correct_answer: str,
    *,
    max_points: float = 0.5,
    numeric_tolerance: float = 0.01,
    remove_spaces: bool = False,
) -> tuple[float, bool]:
    """Return earned points and correctness for a short answer."""
    is_correct = answers_match(
        student_answer,
        correct_answer,
        numeric_tolerance=numeric_tolerance,
        remove_spaces=remove_spaces,
    )
    return (max_points if is_correct else 0.0), is_correct
