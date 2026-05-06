import json
import glob

failed = [
    'exam_de7plus_de01_p1_q01', 
    'exam_de7plus_de03_p1_q01', 
    'exam_de7plus_de03_p1_q09', 
    'exam_de7plus_de04_p1_q01', 
    'exam_de7plus_de04_p1_q02', 
    'exam_de7plus_de05_p1_q02'
]

qs = []
for f in glob.glob('data/exams/*.json'):
    items = json.load(open(f, encoding='utf-8'))
    for q in items:
        if q.get('id') in failed:
            qs.append(q)

for q in qs:
    print(f"ID: {q['id']}")
    print(f"Nội dung: {q['content']}")
    print(f"Đáp án: {q.get('answer')}")
    print(f"Meta: {q.get('metadata')}")
    print("-" * 50)
