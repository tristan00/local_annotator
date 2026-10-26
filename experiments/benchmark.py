"""Measure identical nested 1/3/5 ballot panels; independently repeat 200 records.

The smaller panels are prefixes of five ballots, NOT separate timing runs.
Only full five-ballot panel throughput is measured. No agreement is accuracy.
"""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
import hashlib
import json
import sqlite3
import threading
import time
from local_annotator import Record, run_annotation
from local_annotator.http import LocalChat
from .tasks import TASKS, VERSION, SCHEMAS, prompt_builder
from .common import write_json


def run(root, model, *, workers=8):
    db = sqlite3.connect(root / "ballots.sqlite", check_same_thread=False)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("CREATE TABLE IF NOT EXISTS attempts (model TEXT, task TEXT, repeat INTEGER, record_id TEXT, voter INTEGER, attempt INTEGER, payload TEXT, PRIMARY KEY(model,task,repeat,record_id,voter,attempt))")
    db.execute("CREATE TABLE IF NOT EXISTS records (model TEXT, task TEXT, repeat INTEGER, record_id TEXT, payload TEXT, PRIMARY KEY(model,task,repeat,record_id))")
    if db.execute("SELECT 1 FROM attempts WHERE model=? LIMIT 1", (model,)).fetchone():
        raise RuntimeError("Existing model results: use a fresh run directory, never silently resume")
    lock = threading.Lock()
    adapter = None
    metrics = []
    try:
        for task, fields in TASKS.items():
            adapter = LocalChat("http://127.0.0.1:8091/v1", model, schema=SCHEMAS[task])
            data = [json.loads(line) for line in (root / (task + ".jsonl")).read_text(encoding="utf-8").splitlines()]
            for repeat in (0, 1):
                selected = data if repeat == 0 else data[:200]
                started, completed, failures = time.perf_counter(), 0, 0
                def annotate(row):
                    record = Record(row["id"], row["text"])
                    seed = int.from_bytes(hashlib.sha256(f'{task}:{repeat}:{record.id}'.encode()).digest()[:4], "big") % 1_000_000_000
                    def save(ballot):
                        with lock:
                            db.execute("INSERT INTO attempts VALUES (?,?,?,?,?,?,?)", (model,task,repeat,record.id,ballot.voter,ballot.attempt,json.dumps(asdict(ballot))))
                            db.commit()
                    result = next(run_annotation(None, lambda _: [record], fields, adapter,
                                                 annotators=5, seed=seed, on_ballot=save,
                                                 prompt_builder=prompt_builder(task), output_format="flat"))
                    with lock:
                        db.execute("INSERT INTO records VALUES (?,?,?,?,?)", (model,task,repeat,record.id,json.dumps(asdict(result))))
                        db.commit()
                    return sum(b.error is not None and not b.error.startswith(("ValueError:", "JSONDecodeError:")) for b in result.ballots)
                iterator, pending, aborted = iter(selected), set(), None
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    def submit():
                        row = next(iterator, None)
                        if row is not None:
                            pending.add(pool.submit(annotate, row))
                    for _ in range(workers):
                        submit()
                    while pending:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for future in done:
                            failures += future.result()
                            completed += 1
                        if completed >= 50 and failures / (completed * 5) > 0.2:
                            aborted = "request failure rate exceeded 20% after 50 records"
                        if time.perf_counter() - started > 1800:
                            aborted = "30-minute task/run bound reached"
                        if not aborted:
                            for _ in done:
                                submit()
                        if completed % 100 < len(done):
                            print(model, task, repeat, completed, "/", len(selected), flush=True)
                metric = {"model": model, "task": task, "repeat": repeat, "requested": len(selected),
                          "completed": completed, "seconds": time.perf_counter()-started,
                          "workers": workers, "aborted": aborted, "task_version": VERSION,
                          "ballots_per_record": 5, "temperature": 0.7, "max_output_tokens": 192,
                          "structured_output": "json_schema", "output_format": "flat",
                          "timing": "warm model; includes generation, validation, and synchronous durable writes"}
                metrics.append(metric)
                write_json(root / (model + "-timing.json"), metrics)
                if aborted:
                    raise RuntimeError(aborted)
    finally:
        if adapter is not None:
            adapter.close()
        db.close()
