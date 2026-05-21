import asyncio
import json
from app.db.database import async_session
from app.quiz.service import create_quiz_session

async def run_test():
    async with async_session() as db:
        res = {}
        
        print("Testing derivative_basic...")
        # Lấy 3 câu (trong đó chỉ có 1 câu mẫu trong DB, 2 câu sẽ từ Generator)
        result1 = await create_quiz_session(db, skill_id="derivative_basic", difficulty=3, count=3, user_id=1)
        res["derivative_basic"] = result1
        
        print("Testing function_survey...")
        # Lấy 2 câu (trong đó có 1 câu mẫu trong DB, 1 câu từ Generator)
        result2 = await create_quiz_session(db, skill_id="function_survey", difficulty=3, count=2, user_id=1)
        res["function_survey"] = result2
        
        with open("test_report.json", "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    asyncio.run(run_test())
