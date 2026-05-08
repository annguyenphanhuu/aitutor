import json, pprint, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('data/exams/de7plus_de01.json', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total questions: {len(data)}")
print(f"Keys: {list(data[0].keys())}\n")

for q in data:
    meta = q.get('metadata', {})
    qnum = meta.get('question_number', '')
    print(f"  id={q.get('id')} | q_num={qnum} | type={meta.get('type')} | skill={meta.get('skill_id')} | answer={q.get('answer')}")

print("\n=== Q19 FULL DATA ===")
for q in data:
    meta = q.get('metadata', {})
    if str(meta.get('question_number', '')) == '19':
        pprint.pprint(q)
