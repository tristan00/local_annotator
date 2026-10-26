import argparse
import json
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

from . import serve
from .core import Record, _parse_lines
from .files import annotate_jsonl, vote_jsonl
from .http import LocalChat
from .models import MODELS
from .task import Task

PORT = 8120
WORKERS = 768
GPU = 0.88
MAX_MODEL_LEN = 16384
STARTUP_SECONDS = 1800
NO_LIMIT = 24 * 3600


def log(message):
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=1, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def read_records(path):
    records = [Record(row["id"], row["text"]) for row in rows(path)]
    if len({record.id for record in records}) != len(records):
        raise ValueError(f"{path} has duplicate record ids")
    return records


def adapter(task, name):
    model = MODELS[name]
    return LocalChat(f"http://127.0.0.1:{PORT}/v1", name, model["sampling"], max_tokens=model["max_tokens"], timeout=900,
                     regex=task.regex, messages=task.messages)


def complete(path, records):
    missing = {record.id for record in records} - ({row["id"] for row in rows(path)} if path.exists() else set())
    if missing:
        raise RuntimeError(f"{path} lacks {len(missing)} of {len(records)} records")


def voter_pass(task, name, records, out):
    timing_path = out.with_suffix(".timing.json")
    if timing_path.exists():
        log(f"pass {name} already complete")
        return json.loads(timing_path.read_text(encoding="utf-8"))
    log(f"enter pass {name} records={len(records)}")
    started = time.perf_counter()
    serve.start(name, PORT, GPU, max_model_len=MAX_MODEL_LEN, startup_seconds=STARTUP_SECONDS)
    ready = time.perf_counter()
    try:
        annotate_jsonl(records, out, task.fields, adapter(task, name), task.prompt, workers=WORKERS, seconds=NO_LIMIT,
                       annotators=1, extra_votes=0, retries=1, resume=out.exists())
        annotated = time.perf_counter()
    finally:
        serve.stop(name)
    complete(out, records)
    timing = {"model": name, "records": len(records), "startup_seconds": round(ready - started, 1),
              "annotate_seconds": round(annotated - ready, 1), "stop_seconds": round(time.perf_counter() - annotated, 1),
              "seconds": round(time.perf_counter() - started, 1), "rate": round(len(records) / (annotated - ready), 1)}
    write_json(timing_path, timing)
    log(f"exit pass {timing}")
    return timing


def answers(task, path):
    result, invalid = {}, Counter()
    for row in rows(path):
        ballot = row["ballots"][-1]
        parsed = {}
        if ballot["error"]:
            invalid["error"] += 1
        else:
            parsed, bad = _parse_lines(ballot["raw"], task.fields)
            invalid["field"] += len(bad)
        result[row["id"]] = {field: answer.get("value") for field, answer in parsed.items()}
    return result, invalid


def agreement(one, other, fields):
    same, total = Counter(), Counter()
    for record_id, mine in one.items():
        theirs = other[record_id]
        for field in fields:
            values = [value for value in (mine.get(field), theirs.get(field)) if value is not None]
            total[field] += len(values)
            same[field] += 2 * (len(values) == 2 and values[0] == values[1])
    share = lambda count, of: round(count / of, 3) if of else None
    return {"overall": share(sum(same.values()), sum(total.values())),
            "fields": {field: share(same[field], total[field]) for field in fields if total[field]}}


def quality(task, voters, paths):
    log(f"enter quality voters={voters}")
    started = time.perf_counter()
    parsed = {voter: answers(task, path) for voter, path in zip(voters, paths)}
    report = {"voters": {voter: {"error_ballots": invalid["error"], "invalid_fields": invalid["field"],
                                 "values": sum(value is not None for answer in result.values() for value in answer.values())}
                         for voter, (result, invalid) in parsed.items()},
              "agreement": {f"{one} | {other}": agreement(parsed[one][0], parsed[other][0], task.fields)
                            for one, other in combinations(voters, 2)}}
    log(f"exit quality seconds={time.perf_counter() - started:.1f} {json.dumps(report['voters'])} "
        f"{ {pair: value['overall'] for pair, value in report['agreement'].items()} }")
    return report


def tiebreak(task, records, paths, name, out):
    log(f"enter tiebreak {name}")
    started = time.perf_counter()
    serve.start(name, PORT, GPU, max_model_len=MAX_MODEL_LEN, startup_seconds=STARTUP_SECONDS)
    ready = time.perf_counter()
    try:
        vote_jsonl(records, paths, out, task.fields, adapter(task, name), task.prompt, base=len(paths), workers=WORKERS,
                   seconds=NO_LIMIT, retries=1)
        voted = time.perf_counter()
    finally:
        serve.stop(name)
    complete(out, records)
    labels = rows(out)
    timing = {"model": name, "startup_seconds": round(ready - started, 1), "vote_seconds": round(voted - ready, 1),
              "stop_seconds": round(time.perf_counter() - voted, 1), "seconds": round(time.perf_counter() - started, 1),
              "calls": sum(len(row["ballots"]) > len(paths) for row in labels)}
    log(f"exit tiebreak {timing}")
    return timing, labels


def main():
    parser = argparse.ArgumentParser(prog="local_annotator")
    parser.add_argument("task", type=Path)
    parser.add_argument("records", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.out.exists() and not args.resume:
        parser.error(f"{args.out} exists; pass --resume to continue it")
    started = time.perf_counter()
    log(f"enter run task={args.task} records={args.records} out={args.out}")
    task, records = Task(args.task), read_records(args.records)
    (args.out / "voters").mkdir(parents=True, exist_ok=True)
    paths = [args.out / "voters" / f"{voter}.jsonl" for voter in task.voters]
    passes = [voter_pass(task, voter, records, path) for voter, path in zip(task.voters, paths)]
    measured = quality(task, task.voters, paths)
    tiebreaker, labels = tiebreak(task, records, paths, task.tiebreaker, args.out / "labels.jsonl")
    report = {"task": task.name, "records": len(records), "voters": task.voters, "tiebreaker": task.tiebreaker,
              "passes": passes, "tiebreak": tiebreaker, **measured,
              "unresolved_records": sum(bool(row["unresolved"]) for row in labels),
              "unresolved_fields": dict(Counter(field for row in labels for field in row["unresolved"]).most_common()),
              "resolved_values": dict(Counter(field for row in labels for field in row["values"]).most_common()),
              "resumed": args.resume, "seconds": round(time.perf_counter() - started, 1)}
    write_json(args.out / "report.json", report)
    log(f"exit run seconds={report['seconds']} minutes={report['seconds'] / 60:.1f} tiebreaker_calls={tiebreaker['calls']} "
        f"unresolved_records={report['unresolved_records']} resumed={args.resume}")


if __name__ == "__main__":
    main()
