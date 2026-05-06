import json

math_data = json.load(open('data/math_report_4_exams.json', encoding='utf-8'))
ragas_data = json.load(open('data/ragas_report_4_exams.json', encoding='utf-8'))

math_per_sample = math_data.get('per_sample', [])
ragas_per_sample = ragas_data.get('per_sample', [])

failed_math = []
for s in math_per_sample:
    if s.get('scores', {}).get('accuracy', 1.0) == 0:
        failed_math.append(s)

print("="*60)
print(f"BÁO CÁO CÂU SAI ĐÁP ÁN (SỐ LƯỢNG: {len(failed_math)})")
print("="*60)
for s in failed_math:
    meta = s.get('_meta', {})
    print(f"\nID: {meta.get('id')} | Skill: {meta.get('skill_id')}")
    print(f"Câu hỏi: {s.get('user_input')}")
    print(f"Đáp án đúng: {s.get('reference')}")
    print(f"LLM trả lời: {s.get('response')[:250]}...")
    
print("\n" + "="*60)
print("BÁO CÁO TÌM KIẾM TÀI LIỆU KÉM (CP < 0.5 HOẶC CR < 0.5)")
print("="*60)
low_ragas = []
for s in ragas_per_sample:
    scores = s.get('scores', {})
    cp = scores.get('context_precision', 1.0)
    cr = scores.get('context_recall', 1.0)
    if cp < 0.5 or cr < 0.5:
        low_ragas.append((s, cp, cr))

print(f"Số lượng: {len(low_ragas)}")
for s, cp, cr in low_ragas[:5]:
    meta = s.get('_meta', {})
    print(f"\nID: {meta.get('id')} | Skill: {meta.get('skill_id')} | CP: {cp:.2f} | CR: {cr:.2f}")
    print(f"Câu hỏi: {s.get('user_input')[:150]}...")
    ctx = s.get('retrieved_contexts', [])
    print(f"Số tài liệu tìm được: {len(ctx)}")
