import asyncio
import subprocess
import os

exams = ["de01", "de03", "de04", "de05"]
models = ["o4-mini-2025-04-16"]

os.makedirs("data/evaluation/logs", exist_ok=True)

async def run_eval(exam, model):
    model_safe = model.replace(".", "_").replace("-", "_")
    log_path = f"data/evaluation/logs/eval_{exam}_{model_safe}.log"
    print(f"Starting {exam} with {model}...")
    cmd = ["python", "scripts/run_eval_monitor.py", "--exam", exam, "--solver-model", model, "--save-trace", "--no-ragas"]
    
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    
    with open(log_path, "w", encoding="utf-8") as f:
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=f, stderr=subprocess.STDOUT, env=env
        )
        await process.communicate()
        
    print(f"Finished {exam} with {model}")

async def main():
    sem = asyncio.Semaphore(4)  # run all 4 concurrently
    
    async def worker(exam, model):
        async with sem:
            await run_eval(exam, model)
            
    tasks = []
    for m in models:
        for e in exams:
            tasks.append(worker(e, m))
            
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
