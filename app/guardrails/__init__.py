"""Guardrails — Input validation và Rate limiting.

Hai lớp bảo vệ tại API layer:

1. InputValidator:
   - Prompt injection detection (12 patterns)
   - Message length truncation
   - Empty message check
   - PII masking trong logs (CMND, SĐT, email)

2. RateLimiter:
   - Sliding window per user_id
   - 30 requests/minute, 200 requests/hour
   - In-memory (reset khi restart server — đủ cho production nhỏ)
"""
