# -*- coding: utf-8 -*-
"""Đồng bộ system prompt hardcode trong code lên Langfuse Prompt Management.

Server fetch prompt từ Langfuse (label "production") khi khởi động
(TeacherAgent.__init__), nên khi sửa prompt trong teacher_agent.py mà quên
đẩy lên Langfuse thì bản trên cloud (cũ) sẽ override bản trong code — bug
đã gặp 2026-07-19. Chạy script này sau mỗi lần sửa prompt trong code:

    python scripts/sync_prompts.py           # so sánh + đẩy nếu khác
    python scripts/sync_prompts.py --dry-run # chỉ so sánh, không đẩy

Langfuse giữ version history — có thể rollback trên UI bất cứ lúc nào.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):  # console Windows mặc định cp1258
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.teacher_agent import (  # noqa: E402
    TEACHER_SYSTEM_PROMPT_ANSWER,
    TEACHER_SYSTEM_PROMPT_EXAM,
    TEACHER_SYSTEM_PROMPT_SOCRATIC,
)
from app.utils.langfuse_client import get_langfuse, get_prompt  # noqa: E402

PROMPTS = {
    "aitutor-socratic-v1": TEACHER_SYSTEM_PROMPT_SOCRATIC,
    "aitutor-exam-v1": TEACHER_SYSTEM_PROMPT_EXAM,
    "aitutor-answer-v1": TEACHER_SYSTEM_PROMPT_ANSWER,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="chỉ so sánh, không đẩy")
    args = parser.parse_args()

    client = get_langfuse()
    if client is None:
        print("Langfuse disabled/misconfigured — server sẽ dùng fallback trong code, không cần sync.")
        return 0

    changed = 0
    for name, local in PROMPTS.items():
        remote = get_prompt(name, fallback="")
        if remote.strip() == local.strip():
            print(f"[=] {name}: cloud đã khớp code, bỏ qua")
            continue
        changed += 1
        if args.dry_run:
            print(f"[!] {name}: cloud KHÁC code (cloud {len(remote)} chars, code {len(local)} chars)")
            continue
        client.create_prompt(
            name=name,
            prompt=local,
            labels=["production"],
            type="text",
        )
        print(f"[^] {name}: đã đẩy version mới lên Langfuse (label=production)")

    if args.dry_run and changed:
        print(f"\n{changed} prompt lệch — chạy lại không có --dry-run để đồng bộ.")
    elif not changed:
        print("\nTất cả prompt đã đồng bộ.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
