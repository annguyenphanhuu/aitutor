import sys

with open('app/quiz/service.py', 'r', encoding='utf-8') as f:
    content = f.read()

target = """    if exam_format:
        raw_questions = generate_exam_questions(skill_id, difficulty)
    else:
        raw_questions = generate_questions(skill_id, difficulty, count)

    if not raw_questions:
        return {"error": "Không thể sinh câu hỏi cho kỹ năng này."}"""

replacement = """    from sqlalchemy.sql.expression import func
    
    req_mcq = 3 if exam_format else count
    req_tf = 1 if exam_format else 0
    req_sa = 1 if exam_format else 0
    
    raw_questions = []
    
    async def fetch_bank_questions(q_type, limit):
        if limit <= 0: return []
        res = await db.execute(
            select(QuestionBank)
            .where(QuestionBank.skill_id == skill_id)
            .where(QuestionBank.difficulty == difficulty)
            .where(QuestionBank.question_type == q_type)
            .where(QuestionBank.is_active == True)
            .order_by(func.random())
            .limit(limit)
        )
        return res.scalars().all()

    mcq_bank = await fetch_bank_questions("mcq", req_mcq)
    tf_bank = await fetch_bank_questions("true_false", req_tf) if req_tf else []
    sa_bank = await fetch_bank_questions("short_answer", req_sa) if req_sa else []

    bank_questions = mcq_bank + tf_bank + sa_bank
    
    for bq in bank_questions:
        raw_questions.append({
            "skill_id": bq.skill_id,
            "skill_ids": [bq.skill_id],
            "difficulty": bq.difficulty,
            "question_type": bq.question_type,
            "question_latex": bq.question_latex,
            "choices": bq.choices,
            "correct_index": bq.correct_index,
            "statements": bq.statements,
            "correct_answer": bq.correct_answer,
            "points": bq.points,
            "explanation": bq.explanation,
            "sympy_expr": "",
        })
    
    missing = (req_mcq + req_tf + req_sa) - len(raw_questions)
    if missing > 0:
        if exam_format:
            gen_qs = generate_exam_questions(skill_id, difficulty)
        else:
            gen_qs = generate_questions(skill_id, difficulty, missing)
        raw_questions.extend(gen_qs)

    if not raw_questions:
        return {"error": "Không thể sinh câu hỏi cho kỹ năng này."}"""

target_crlf = target.replace('\n', '\r\n')
if target in content:
    content = content.replace(target, replacement)
elif target_crlf in content:
    content = content.replace(target_crlf, replacement)
else:
    print('Failed to replace target')
    sys.exit(1)

with open('app/quiz/service.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated successfully")
