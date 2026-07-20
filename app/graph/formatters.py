"""Text formatters + template tĩnh cho các node — copy nguyên văn từ Orchestrator.

Giữ từng byte giống output cũ để frontend không phân biệt được engine nào
đang chạy (parity contract). Khi Orchestrator cũ bị xóa (Phase 6), đây là
nguồn duy nhất của các text này.
"""

from __future__ import annotations

from app.agents.contracts import PedagogyAssessment

OFF_TOPIC_TEXT = (
    "Phần này nằm ngoài chuyên môn của thầy mất rồi — thầy là gia sư Toán 12, "
    "nên chỉ giúp em tốt nhất được trong phạm vi Toán và việc ôn thi thôi nhé. 😊\n\n"
    "Khi nào em cần giải thích một bài Toán, luyện quiz, hay lập kế hoạch ôn tập, "
    "cứ nhắn thầy — mình bắt đầu ngay. Em đang học đến phần nào rồi?"
)

# Fallback cho node social khi LLM lỗi — bình thường social sinh text cá nhân hóa.
GREETING_FALLBACK_TEXT = (
    "Chào em! 😊 Thầy là gia sư Toán 12 của em đây.\n\n"
    "Thầy có thể giải thích bài em chưa hiểu, cho em luyện quiz, "
    "lập kế hoạch ôn tập, hoặc làm bài chẩn đoán để biết em đang ở đâu.\n\n"
    "Em muốn bắt đầu từ phần nào?"
)

MOTIVATION_FALLBACK_TEXT = (
    "Thầy hiểu cảm giác đó của em — ai ôn thi cũng có lúc thấy quá tải, "
    "và điều đó không có nghĩa là em kém. 💪\n\n"
    "Mất gốc hay điểm thấp bây giờ đều kịp cải thiện nếu mình đi từng bước nhỏ. "
    "Em làm thử bài **chẩn đoán năng lực** để thầy biết em hổng ở đâu, "
    "rồi thầy lập kế hoạch ôn vừa sức cho em nhé?"
)

REVIEW_TEXT = (
    "📖 **Ôn tập chống quên lãng**\n\n"
    "Hệ thống sẽ kiểm tra các kiến thức em có nguy cơ quên.\n"
    "Sử dụng nút **📖 Ôn tập** hoặc gọi API: `GET /api/review/due` để lấy thẻ ôn tập.\n\n"
    "💡 Thuật toán SM-2 (SuperMemo) sẽ tự lên lịch ôn tập cá nhân hóa!"
)

DIAGNOSTIC_TEXT = (
    "🔍 **Test Chẩn Đoán Năng Lực**\n\n"
    "Bài test này gồm 12 câu hỏi từ tất cả các chương, "
    "giúp đánh giá điểm mạnh/yếu của em.\n\n"
    "Sử dụng nút **🔍 Chẩn đoán** hoặc gọi API: `POST /api/diagnostic/start`.\n\n"
    "⏱️ Thời gian dự kiến: khoảng 10-15 phút."
)


def format_quiz_text(skill_name: str, target_skill: str, difficulty: int) -> str:
    return (
        f"📝 **Bài kiểm tra: {skill_name}**\n\n"
        f"Được em! Thầy mở bài quiz cho em ngay bên dưới — độ khó sẽ tự điều chỉnh "
        f"theo trình độ hiện tại của em.\n\n"
        f"💡 Sau mỗi câu thầy sẽ chấm ngay và cập nhật hồ sơ năng lực cho em. Cố lên nhé! 💪"
    )


def format_plan(plan: dict) -> str:
    """Format study plan into readable text."""
    lines = [f"📋 **{plan.get('summary', 'Kế hoạch học tập')}**\n"]

    priorities = plan.get("priorities", [])
    for i, p in enumerate(priorities, 1):
        urgency_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(p.get("urgency", "medium"), "⚪")
        lines.append(
            f"{i}. {urgency_emoji} **{p.get('skill_name', '')}**: "
            f"{p.get('action', '')} ({p.get('exercises', 5)} bài)"
        )

    encouragement = plan.get("encouragement", "")
    if encouragement:
        lines.append(f"\n💪 {encouragement}")

    return "\n".join(lines)


HEDGED_NOTE = (
    "\n\n⚠️ *Kết quả chấm chưa được kiểm chứng tự động — "
    "hồ sơ năng lực của em chưa được cập nhật từ câu này.*"
)


def format_assessment(assessment: PedagogyAssessment, hedged: bool = False) -> str:
    """Format assessment result into readable text (port từ _format_assessment)."""
    emoji = "✅" if assessment.is_correct else "❌"
    score = assessment.score

    lines = [
        f"{emoji} **Điểm: {score:.0%}**\n",
        f"**Nhận xét:** {assessment.feedback}\n",
    ]

    if not assessment.is_correct:
        error_labels = {
            "calculation": "Lỗi tính toán",
            "conceptual": "Hiểu sai khái niệm",
            "procedural": "Sai phương pháp giải",
        }
        error_type = assessment.error_type
        lines.append(f"**Loại lỗi:** {error_labels.get(error_type, error_type)}\n")

    lines.append(f"**Lời giải đúng:**\n{assessment.correct_solution}")

    text = "\n".join(lines)
    if hedged:
        text += HEDGED_NOTE
    return text
