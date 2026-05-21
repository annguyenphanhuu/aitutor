import asyncio
import json
import os
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.database import async_session, engine
from app.db.models import Base, QuestionBank

async def init_db():
    async with engine.begin() as conn:
        # Create tables if not exist
        await conn.run_sync(Base.metadata.create_all)

async def seed_questions():
    await init_db()
    
    sample_questions = [
        {
            "skill_id": "derivative_basic",
            "cognitive_level": "van_dung",
            "difficulty": 3,
            "question_type": "mcq",
            "question_latex": "Một vật chuyển động có phương trình $s(t) = t^3 - 3t^2 + 5t + 2$. Tính vận tốc của vật tại thời điểm gia tốc bị triệt tiêu.",
            "choices": [
                "$v = 2$",
                "$v = 5$",
                "$v = 1$",
                "$v = -1$"
            ],
            "correct_index": 0,
            "points": 0.25,
            "explanation": "Gia tốc $a(t) = s''(t) = 6t - 6$. Gia tốc triệt tiêu khi $6t - 6 = 0 \Rightarrow t = 1$. Vận tốc $v(t) = s'(t) = 3t^2 - 6t + 5$. Tại $t = 1$, $v(1) = 3(1)^2 - 6(1) + 5 = 2$.",
            "source": "Đề thi thử THPT QG"
        },
        {
            "skill_id": "function_survey",
            "cognitive_level": "van_dung_cao",
            "difficulty": 3,
            "question_type": "mcq",
            "question_latex": "Một doanh nghiệp dự kiến chi phí sản xuất $x$ sản phẩm là $C(x) = x^3 - 6x^2 + 15x + 100$ (triệu đồng). Tìm số sản phẩm $x$ để chi phí biên nhỏ nhất.",
            "choices": [
                "$x = 2$",
                "$x = 3$",
                "$x = 1$",
                "$x = 6$"
            ],
            "correct_index": 0,
            "points": 0.25,
            "explanation": "Chi phí biên $C'(x) = 3x^2 - 12x + 15$. Để chi phí biên nhỏ nhất thì hàm bậc hai $C'(x)$ đạt min tại $x = -b/2a = 12/(2*3) = 2$.",
            "source": "Sách Toán thực tế 12"
        }
    ]

    async with async_session() as db:
        for q in sample_questions:
            db_q = QuestionBank(**q)
            db.add(db_q)
        
        await db.commit()
        print(f"Đã thêm thành công {len(sample_questions)} câu hỏi Toán thực tế vào QuestionBank!")

if __name__ == "__main__":
    asyncio.run(seed_questions())
