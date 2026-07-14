"""Deterministic answer-tag instructions and post-processing for exam output."""

from __future__ import annotations

import re
from typing import Optional


_ANSWER_TAG_RE = re.compile(r"<answer\b[^>]*>(.*?)</answer\s*>", re.IGNORECASE | re.DOTALL)


def normalize_question_type(question_type: str | None) -> str:
    """Map dataset and splitter aliases to one canonical question type."""
    value = (question_type or "").strip().lower()
    aliases = {
        "exam_mcq": "mcq",
        "multiple_choice": "mcq",
        "exam_true_false": "true_false",
        "true-false": "true_false",
        "tf": "true_false",
        "exam_short_answer": "short_answer",
        "short": "short_answer",
    }
    return aliases.get(value, value)


def answer_format_instruction(question_type: str | None) -> str:
    """Return a strict runtime instruction for the requested exam output."""
    qtype = normalize_question_type(question_type)
    if qtype == "mcq":
        return (
            "ĐỊNH DẠNG ĐÁP ÁN BẮT BUỘC: Dòng đầu tiên phải là "
            "<answer>X</answer>, trong đó X chỉ là A, B, C hoặc D. "
            "Sau đó mới trình bày lời giải. Chỉ xuất hiện đúng một thẻ <answer>."
        )
    if qtype == "true_false":
        return (
            "ĐỊNH DẠNG ĐÁP ÁN BẮT BUỘC: Dòng đầu tiên phải có dạng "
            "<answer>a-T,b-F,c-T,d-F</answer> (T=Đúng, F=Sai), đủ bốn ý "
            "a/b/c/d. Sau đó mới giải thích. Chỉ xuất hiện đúng một thẻ <answer>."
        )
    if qtype == "short_answer":
        return (
            "ĐỊNH DẠNG ĐÁP ÁN BẮT BUỘC: Dòng đầu tiên phải là "
            "<answer>GIÁ_TRỊ</answer>; bên trong chỉ chứa số hoặc phân số đã làm "
            "tròn đúng yêu cầu, không kèm đơn vị hay lời giải thích."
        )
    return ""


def _extract_mcq(text: str) -> Optional[str]:
    tagged = _ANSWER_TAG_RE.search(text)
    if tagged:
        match = re.search(r"\b([A-D])\b", tagged.group(1), re.IGNORECASE)
        if match:
            return match.group(1).upper()

    patterns = (
        r"(?:đáp\s*án(?:\s+đúng)?|kết\s*quả)\s*(?:là|:|-)?\s*[*_`]*\(?([A-D])\)?\b",
        r"(?:chọn|chọn\s+phương\s+án|phương\s+án)\s*(?:là|:|-)?\s*[*_`]*\(?([A-D])\)?\b",
    )
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, text, flags=re.IGNORECASE))
    return matches[-1].upper() if matches else None


def _extract_true_false(text: str) -> Optional[str]:
    tagged = _ANSWER_TAG_RE.search(text)
    source = tagged.group(1) if tagged else text
    pairs = re.findall(
        r"\b([a-d])\s*[-:=.)]?\s*(T|F|Đ|S|đúng|sai)\b",
        source,
        flags=re.IGNORECASE,
    )
    values: dict[str, str] = {}
    for label, value in pairs:
        canonical = "T" if value.lower() in {"t", "đ", "đúng"} else "F"
        values[label.lower()] = canonical
    if set(values) != {"a", "b", "c", "d"}:
        return None
    return ",".join(f"{label}-{values[label]}" for label in "abcd")


_NUMBER_RE = re.compile(
    r"(?<![\w.])([+-]?(?:\d+(?:[.,]\d+)?|\d+\s*/\s*\d+))(?![\w.])"
)


def _extract_short_answer(text: str) -> Optional[str]:
    tagged = _ANSWER_TAG_RE.search(text)
    if tagged:
        match = _NUMBER_RE.fullmatch(tagged.group(1).strip())
        if match:
            return match.group(1).replace(" ", "").replace(",", ".")

    matches = re.findall(
        r"(?:đáp\s*án|kết\s*quả)\s*(?:là|:|-)?\s*"
        r"([+-]?(?:\d+(?:[.,]\d+)?|\d+\s*/\s*\d+))",
        text,
        flags=re.IGNORECASE,
    )
    if not matches:
        return None
    return matches[-1].replace(" ", "").replace(",", ".")


def ensure_answer_tag(response: str, question_type: str | None) -> str:
    """Canonicalize a model answer to exactly one leading ``<answer>`` tag.

    The function only repairs output when the answer can be extracted
    unambiguously; it never guesses a missing answer from option text.
    """
    text = str(response or "").strip()
    qtype = normalize_question_type(question_type)
    extractors = {
        "mcq": _extract_mcq,
        "true_false": _extract_true_false,
        "short_answer": _extract_short_answer,
    }
    extractor = extractors.get(qtype)
    if not extractor:
        return text

    answer = extractor(text)
    if not answer:
        return text

    explanation = _ANSWER_TAG_RE.sub("", text).strip()
    canonical = f"<answer>{answer}</answer>"
    return f"{canonical}\n\n{explanation}" if explanation else canonical

