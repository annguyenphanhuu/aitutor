"""Text formatters + template tĩnh cho các node — copy nguyên văn từ Orchestrator.

Giữ từng byte giống output cũ để frontend không phân biệt được engine nào
đang chạy (parity contract). Khi Orchestrator cũ bị xóa (Phase 6), đây là
nguồn duy nhất của các text này.
"""

from __future__ import annotations

from app.agents.contracts import PedagogyAssessment

OFF_TOPIC_TEXT = (
    "Thầy hiểu em đang có chuyện bên ngoài, và điều đó hoàn toàn bình thường. 😊\n\n"
    "Nhưng thầy là gia sư Toán 12 — thầy chỉ có thể đồng hành cùng em trên con đường chinh phục Toán thôi nhé.\n\n"
    "Nếu em đang căng thẳng vì học hành, thầy rất sẵn sàng giúp em:\n"
    "- 📋 Lập **kế hoạch ôn tập** phù hợp để bớt áp lực\n"
    "- 🔍 **Chẩn đoán năng lực** để biết mình đang ở đâu\n"
    "- 💡 Giải thích những phần Toán em chưa hiểu\n\n"
    "Em muốn bắt đầu từ đâu?"
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
        f"📝 **Bài kiểm tra: {skill_name}** (Độ khó: {difficulty})\n\n"
        f"Sử dụng nút **📝 Làm quiz** trên giao diện để bắt đầu, "
        f"hoặc gọi API: `POST /api/quiz/generate` với skill_id=`{target_skill}`.\n\n"
        f"💡 Mình sẽ tự động đánh giá và cập nhật năng lực của em sau mỗi câu hỏi!"
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
