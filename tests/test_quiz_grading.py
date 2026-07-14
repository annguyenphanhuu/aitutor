"""Tests for shared THPT quiz/exam scoring rules."""

import pytest

from app.quiz.grading import answers_match, score_short_answer, score_true_false


@pytest.mark.parametrize(
    ("num_correct", "expected_points"),
    [(0, 0.0), (1, 0.1), (2, 0.25), (3, 0.5), (4, 1.0)],
)
def test_true_false_score_table(num_correct, expected_points):
    correct = [True, True, True, True]
    student = [True] * num_correct + [False] * (4 - num_correct)
    assert score_true_false(student, correct) == (expected_points, num_correct)


def test_short_answer_accepts_decimal_comma_with_practice_tolerance():
    assert score_short_answer("1,005", "1.0") == (0.5, True)


def test_short_answer_tolerance_is_strict():
    assert score_short_answer("1.01", "1.0") == (0.0, False)


def test_exam_style_numeric_comparison_is_exact_but_normalized():
    assert answers_match(" 1,0 ", "1", numeric_tolerance=0.0, remove_spaces=True)
    assert not answers_match("1.001", "1", numeric_tolerance=0.0, remove_spaces=True)


def test_text_comparison_is_case_insensitive():
    assert answers_match(" ĐÚNG ", "đúng")
