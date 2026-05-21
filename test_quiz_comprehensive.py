import asyncio
import json
from app.db.database import async_session
from app.quiz.service import create_quiz_session, submit_quiz_answer
import traceback

async def run_comprehensive_test():
    results = {}
    
    async with async_session() as db:
        try:
            print("--- Test 1: Standard Practice (derivative_basic) ---")
            # Yêu cầu 5 câu, trong DB có 1 câu mẫu, SymPy sinh thêm 4 câu
            session1 = await create_quiz_session(db, skill_id="derivative_basic", difficulty=3, count=5, user_id=1)
            results["Test_1_Standard_Practice"] = session1
            
            print("--- Test 2: Exam Format (function_survey) ---")
            # Yêu cầu exam_format (3 MCQ, 1 TF, 1 SA). DB có 1 MCQ. Fallback sinh 2 MCQ, 1 TF, 1 SA.
            session2 = await create_quiz_session(db, skill_id="function_survey", difficulty=3, exam_format=True, user_id=1)
            results["Test_2_Exam_Format"] = session2
            
            print("--- Test 3: Submitting Answers Simulation ---")
            submit_results = []
            session_id = session1["session_id"]
            questions = session1["questions"]
            
            for idx, q in enumerate(questions):
                q_id = q["id"]
                q_type = q["question_type"]
                print(f"Submitting answer for Q{idx+1} (ID: {q_id}, Type: {q_type})...")
                
                # Giả lập học sinh luôn chọn index 0 cho MCQ
                ans = {"session_id": session_id, "question_id": q_id}
                if q_type == "mcq":
                    ans["selected_index"] = 0
                elif q_type == "true_false":
                    ans["tf_answers"] = [True, False, True, False]
                elif q_type == "short_answer":
                    ans["text_answer"] = "1.0"
                    
                # Gọi service
                res = await submit_quiz_answer(
                    db,
                    session_id=session_id,
                    question_id=q_id,
                    selected_index=ans.get("selected_index"),
                    tf_answers=ans.get("tf_answers"),
                    text_answer=ans.get("text_answer"),
                    user_id=1
                )
                submit_results.append({
                    "question_id": q_id,
                    "submitted_answer": ans,
                    "grading_result": res
                })
            
            results["Test_3_Grading"] = submit_results
            
            # Lưu file json
            with open("comprehensive_report.json", "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
                
            print("Comprehensive tests completed successfully!")

        except Exception as e:
            print("Test failed with exception:")
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_comprehensive_test())
