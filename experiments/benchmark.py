"""Bounded, resumable benchmark; all attempts durably persisted in SQLite."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
try:
    from .paths import ARTIFACT_ROOT, DATA, MEASUREMENTS
except ImportError:
    from paths import ARTIFACT_ROOT, DATA, MEASUREMENTS

import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_annotator import AttemptStore, OpenAIAdapter, run_annotation
from experiments.tasks import TASKS, VERSION


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--limit", type=int, default=10000)
    p.add_argument("--voters", type=int, default=5)
    p.add_argument("--repeat", type=int, default=0)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--label", default="main")
    p.add_argument("--task", choices=["age", "appeal", "both"], default="both")
    p.add_argument("--url", default="http://127.0.0.1:8091/v1/chat/completions")
    args = p.parse_args()
    if min(args.limit, args.voters, args.workers) < 1 or args.repeat < 0:
        p.error("limit, voters and workers must be positive; repeat must be nonnegative")
    MEASUREMENTS.mkdir(parents=True, exist_ok=True)
    model = OpenAIAdapter(args.url, args.model, temperature=0.7, max_tokens=64)
    tasks = TASKS if args.task == "both" else {args.task: TASKS[args.task]}
    with model, AttemptStore(MEASUREMENTS / "attempts.sqlite") as store:
        for task, fields in tasks.items():
            path = DATA / (task + ".jsonl")
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()][:args.limit]
            if not rows or len({r['id'] for r in rows}) != len(rows):
                raise ValueError('Benchmark input must be nonempty and contain unique record IDs')
            run_id = f"{args.label}-{args.model}-{task}-n{len(rows)}-v{args.voters}-r{args.repeat}"
            config = {**vars(args), "task": task, "task_version": VERSION,
                      "fields": {k: asdict(v) for k,v in fields.items()}, "settings": model.config,
                      "sample_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            config = json.loads(json.dumps(config))  # Canonicalize tuple fields for resume comparison.
            existing = store.connection.execute("SELECT config FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if existing is None:
                store.start_run(config, run_id)
            elif json.loads(existing[0]) != config:
                raise ValueError("Resume config differs")
            done = {json.loads(row[0]) for row in store.connection.execute("SELECT record_id FROM results WHERE run_id=?", (run_id,))}
            todo = [(i,r) for i,r in enumerate(rows) if r["id"] not in done]
            sink = store.attempt_sink(run_id)
            start = time.time()
            def process(item):
                i, record = item
                result = next(run_annotation(None, lambda _: [record], fields, model, args.voters,
                    seed=20261025 + args.repeat * 1000000 + i * 10, task_version=VERSION,
                    timeout=45, retries=0, concurrency=1, on_attempt=sink))
                store.save_result(run_id, result)
                return result
            count = failed = 0
            recent_failures = deque(maxlen=20)
            print(f"START {run_id} todo={len(todo)}", flush=True)
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                remaining = iter(todo)
                futures = set()
                for _ in range(min(args.workers, len(todo))):
                    futures.add(pool.submit(process, next(remaining)))
                while futures:
                    completed, futures = wait(futures, return_when=FIRST_COMPLETED)
                    for future in completed:
                        result = future.result()
                        count += 1
                        failed += sum(a["status"] == "failed" for b in result["ballots"] for a in b["attempts"])
                        recent_failures.append(all(a['status'] == 'failed' for b in result['ballots'] for a in b['attempts']))
                        if count % 100 == 0:
                            print(f"{run_id} records={count}/{len(todo)} seconds={time.time()-start:.1f} failed={failed}", flush=True)
                        if (len(recent_failures) == 20 and sum(recent_failures) >= 15) or (count >= min(100, len(todo)) and failed > args.voters * count / 2):
                            for f in futures:
                                f.cancel()
                            raise RuntimeError("Failure rate exceeded 50%; stopped submitting work")
                        item = next(remaining, None)
                        if item is not None:
                            futures.add(pool.submit(process, item))
            session = {"run_id": run_id, "started_epoch": start, "seconds": time.time()-start,
                       "completed_this_session": count, "previously_completed": len(done), "failed_attempts": failed}
            with (MEASUREMENTS / "sessions.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(session) + "\n")
            print("DONE " + json.dumps(session), flush=True)


if __name__ == "__main__":
    main()
