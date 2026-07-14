"""Tests for deterministic exam answer formatting."""

from app.utils.answer_format import answer_format_instruction, ensure_answer_tag


def test_mcq_existing_tag_is_canonicalized_once_at_start():
    response = "Giải nhanh. <answer>b</answer>\nĐáp án đúng là B."
    result = ensure_answer_tag(response, "exam_mcq")

    assert result.startswith("<answer>B</answer>")
    assert result.lower().count("<answer>") == 1


def test_mcq_missing_tag_is_repaired_from_explicit_conclusion():
    result = ensure_answer_tag("Suy ra đáp án đúng là B.", "mcq")
    assert result.startswith("<answer>B</answer>")


def test_mcq_does_not_guess_from_option_text_only():
    response = "A. 1\nB. 2\nC. 3\nD. 4"
    assert ensure_answer_tag(response, "mcq") == response


def test_true_false_is_canonicalized_in_label_order():
    response = "Kết luận: d sai, b đúng, a đúng, c sai."
    result = ensure_answer_tag(response, "true_false")
    assert result.startswith("<answer>a-T,b-T,c-F,d-F</answer>")


def test_short_answer_strips_unit_from_tag_contract():
    result = ensure_answer_tag("Kết quả là 2,50 mét.", "exam_short_answer")
    assert result.startswith("<answer>2.50</answer>")


def test_instruction_is_explicit_for_mcq():
    instruction = answer_format_instruction("mcq")
    assert "<answer>X</answer>" in instruction
    assert "đúng một thẻ" in instruction

