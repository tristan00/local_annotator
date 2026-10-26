from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import threading
import time
from .core import run_annotation

PARSE_ERRORS = ("ValueError:", "JSONDecodeError:")


def unresolved(result):
    return any(decision["status"] == "unresolved" for decision in result.decisions.values())


def annotate_jsonl(records, out, fields, model, prompt_builder, *, workers, seconds, output_format="lines", annotators=3,
                   extra_votes=3, resume=False, report_every=5000, seed=0):
    started = time.perf_counter()
    out = Path(out)
    if out.exists() and not resume:
        raise RuntimeError(f"{out} exists; pass resume to continue it")
    done = {json.loads(line)["id"] for line in out.open(encoding="utf-8")} if out.exists() else set()
    todo = [record for record in records if record.id not in done]
    print(f"start {out} done={len(done)} todo={len(todo)} workers={workers} seconds={seconds}", flush=True)
    lock, counts, voters = threading.Lock(), Counter(), Counter()

    def annotate(record):
        if time.perf_counter() - started > seconds:
            return
        base = int.from_bytes(hashlib.sha256(f"{seed}:{record.id}".encode()).digest()[:4], "big") % 1_000_000_000
        result = next(run_annotation(None, lambda _: [record], fields, model, annotators, extra_votes=extra_votes,
                                     disagreement=unresolved, seed=base, prompt_builder=prompt_builder,
                                     output_format=output_format))
        failed = [ballot.error for ballot in result.ballots if ballot.error and not ballot.error.startswith(PARSE_ERRORS)]
        if failed:
            raise RuntimeError(f"{record.id}: {failed[0]}")
        values = {name: decision["answer"]["value"] for name, decision in result.decisions.items()
                  if decision["status"] == "resolved" and decision["answer"]["status"] == "value"}
        row = {"id": record.id, "values": values,
               "unresolved": sorted(name for name, decision in result.decisions.items() if decision["status"] == "unresolved"),
               "ballots": [{"voter": ballot.voter, "raw": ballot.raw, "error": ballot.error, "invalid": ballot.invalid}
                           for ballot in result.ballots]}
        with lock:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            voters[len({ballot.voter for ballot in result.ballots})] += 1
            counts.update(records=1, values=len(values), unresolved=len(row["unresolved"]), ballots=len(result.ballots),
                          invalid=sum(bool(ballot.error or ballot.invalid) for ballot in result.ballots))
            if counts["records"] % report_every == 0:
                elapsed = time.perf_counter() - started
                print(f"progress {dict(counts)} voters={dict(sorted(voters.items()))} seconds={elapsed:.0f} "
                      f"rate={counts['records'] / elapsed:.1f}/s", flush=True)

    with out.open("a", encoding="utf-8") as stream, ThreadPoolExecutor(workers) as pool:
        try:
            for _ in pool.map(annotate, todo):
                pass
        except BaseException:
            pool.shutdown(cancel_futures=True)
            raise
    elapsed = time.perf_counter() - started
    print(f"exit {out} {dict(counts)} voters={dict(sorted(voters.items()))} remaining={len(todo) - counts['records']} "
          f"seconds={elapsed:.0f} rate={counts['records'] / max(elapsed, 1e-9):.1f}/s", flush=True)
    return counts
