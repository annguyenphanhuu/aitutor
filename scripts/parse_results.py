import os
import glob
import re

logs = glob.glob("data/evaluation/logs/*.log")
results = {}

for log in logs:
    basename = os.path.basename(log)
    # Parse exam and model
    m = re.match(r"eval_(de\d+)_(.*)\.log", basename)
    if not m:
        continue
    exam, model = m.group(1), m.group(2)
    
    with open(log, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find summary block
    summary_match = re.search(r"SUMMARY — (\d+) câu.*?Overall Accuracy\s*:\s*([\d.]+).*?LLMExaminer Weighted Avg\s*:\s*([\d.]+)", content, re.DOTALL)
    
    if summary_match:
        questions, accuracy, llm_examiner = summary_match.groups()
        if model not in results:
            results[model] = {}
        results[model][exam] = {
            "questions": int(questions),
            "accuracy": float(accuracy),
            "llm_examiner": float(llm_examiner)
        }

print("| Model | Exam | Accuracy | LLM Examiner Score |")
print("|-------|------|----------|--------------------|")
for model in sorted(results.keys()):
    for exam in sorted(results[model].keys()):
        d = results[model][exam]
        print(f"| {model} | {exam} | {d['accuracy']*100:.1f}% | {d['llm_examiner']:.3f} |")
