"""Quantitative report from stored evidence, never from model-generated labels."""
from collections import Counter, defaultdict
import csv
import itertools
import json
import math
import sqlite3
import statistics
from .common import write_json


def panel(ballots, field, size):
    selected = [b for b in ballots if b["voter"] < size]
    answers = [json.dumps(b["answers"][field], sort_keys=True) for b in selected if field in b["answers"]]
    counts = Counter(answers)
    winner, support = counts.most_common(1)[0] if counts else (None, 0)
    return winner if support > size // 2 else None, answers


def agreement(left, right):
    common = set(left) & set(right)
    comparable = [rid for rid in common if left[rid] is not None and right[rid] is not None]
    same = sum(left[rid] == right[rid] for rid in comparable)
    return {"common_records": len(common), "both_resolved": len(comparable), "same": same,
            "agreement_among_both_resolved": same / len(comparable) if comparable else None,
            "same_over_all_common": same / len(common) if common else None}


def generate(root):
    groups = defaultdict(list)
    if (root / "ballots.sqlite").exists():
        with sqlite3.connect(f"file:{(root / 'ballots.sqlite').as_posix()}?mode=ro",uri=True) as db:
            if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='records'").fetchone():
                for model, task, repeat, rid, payload in db.execute("SELECT model,task,repeat,record_id,payload FROM records"):
                    groups[model,task,repeat].append(json.loads(payload))
    rows, decisions, qualitative = [], {}, []
    for (model, task, repeat), records in sorted(groups.items()):
        for size in (1,3,5):
            selected, valid, failed, invalid, pairs, equal_pairs, unanimous = [], 0, 0, 0, 0, 0, 0
            labels, resolved = Counter(), {}
            for record in records:
                ballots = [b for b in record["ballots"] if b["voter"] < size]
                selected.extend(ballots)
                winner, answers = panel(ballots, task, size)
                resolved[record["record_id"]] = winner
                valid += len(answers)
                for ballot in ballots:
                    if task not in ballot["answers"]:
                        if ballot["error"] and not ballot["error"].startswith(("JSONDecodeError:", "ValueError:")):
                            failed += 1
                        else:
                            invalid += 1
                pair_values = list(itertools.combinations(answers, 2))
                pairs += len(pair_values)
                equal_pairs += sum(a == b for a,b in pair_values)
                unanimous += len(answers) == size and len(set(answers)) == 1
                if winner:
                    labels[winner] += 1
                if size == 5 and repeat == 0 and len(qualitative) < 30 and len(set(answers)) > 1:
                    qualitative.append({"model":model,"task":task,"record_id":record["record_id"],"counts":dict(Counter(answers))})
            latency = sorted(b["seconds"] for b in selected)
            decisions[model,task,repeat,size] = resolved
            n = len(records)
            rows.append({"model":model,"task":task,"repeat":repeat,"voters":size,"records":n,
                "ballots":len(selected),"valid_ballots":valid,"invalid_ballots":invalid,"failed_ballots":failed,
                "resolved":sum(x is not None for x in resolved.values()),
                "unresolved":sum(x is None for x in resolved.values()),
                "unanimous":unanimous,"valid_pairs":pairs,"equal_pairs":equal_pairs,
                "pair_agreement":equal_pairs/pairs if pairs else None,
                "median_ballot_seconds":statistics.median(latency),
                "p95_ballot_seconds":latency[max(0,math.ceil(len(latency)*.95)-1)],
                "label_distribution":dict(labels)})
    repeated, between = [], []
    for model,task,repeat,size in decisions:
        if repeat == 0 and (model,task,1,size) in decisions:
            repeated.append({"model":model,"task":task,"voters":size,
                **agreement(decisions[model,task,0,size],decisions[model,task,1,size])})
    for task in ("age","appeal"):
        models = sorted({m for m,t,r,s in decisions if t==task and r==0 and s==5})
        for a,b in itertools.combinations(models,2):
            between.append({"model_a":a,"model_b":b,"task":task,"voters":5,
                **agreement(decisions[a,task,0,5],decisions[b,task,0,5])})
    write_json(root / "metrics.json", {"panels":rows,"repeat_consistency":repeated,"between_models":between})
    write_json(root / "disagreement-examples.json", qualitative)
    with (root / "metrics.csv").open("w", newline="",encoding="utf-8") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows({**r,"label_distribution":json.dumps(r["label_distribution"])} for r in rows)
    status = json.loads((root / "matrix-status.json").read_text())
    lines = ["# Local Annotator: fresh local-model comparison", "",
        "Status is provisional while any model is loading or benchmarking. Completed-record denominators below can be smaller than the requested sample; consult matrix-status.json and task timing files for completion.", "",
        "## Coverage", "", "| Model | Status | Cold start seconds | Weight GB |", "|---|---|---:|---:|"]
    for name, meta in status.items():
        lines.append(f'| {name} | {meta["status"]} | {meta.get("cold_start_seconds",0):.1f} | {meta.get("weights_bytes",0)/1e9:.2f} |')
        if meta.get("error"):
            lines.append(f'\n{name}: {meta["error"]}\n')
    lines += ["", "## Main sample", "", "| Model | Task | Voters | N | Resolved | Unresolved | Valid ballots / all | Invalid | Failed | Valid-pair agreement |",
              "|---|---|---:|---:|---:|---:|---|---:|---:|---:|"]
    for r in rows:
        if r["repeat"] != 0:
            continue
        pair = f'{r["pair_agreement"]:.3f}' if r["pair_agreement"] is not None else "N/A"
        lines.append(f'| {r["model"]} | {r["task"]} | {r["voters"]} | {r["records"]} | {r["resolved"]} | {r["unresolved"]} | {r["valid_ballots"]}/{r["ballots"]} | {r["invalid_ballots"]} | {r["failed_ballots"]} | {pair} |')
    lines += ["", "## Warm throughput and latency", "", "| Model | Task | Records/s (five votes each) | Median ballot s | p95 ballot s |",
              "|---|---|---:|---:|---:|"]
    for timing_path in sorted(root.glob("*-timing.json")):
        for timing in json.loads(timing_path.read_text()):
            if timing["repeat"] != 0:
                continue
            row = next(r for r in rows if r["model"]==timing["model"] and r["task"]==timing["task"] and r["repeat"]==0 and r["voters"]==5)
            lines.append(f'| {row["model"]} | {row["task"]} | {timing["completed"]/timing["seconds"]:.2f} | {row["median_ballot_seconds"]:.3f} | {row["p95_ballot_seconds"]:.3f} |')
    lines += ["", "## Repeat consistency (five-voter results)", "", "| Model | Task | Common records | Both resolved | Same | Same / both resolved |",
              "|---|---|---:|---:|---:|---:|"]
    for r in repeated:
        if r["voters"] == 5:
            score = f'{r["agreement_among_both_resolved"]:.3f}' if r["agreement_among_both_resolved"] is not None else "N/A"
            lines.append(f'| {r["model"]} | {r["task"]} | {r["common_records"]} | {r["both_resolved"]} | {r["same"]} | {score} |')
    lines += ["", "## Between-model agreement (five-voter results)", "", "| Model A | Model B | Task | Common | Both resolved | Same | Same / both resolved |",
              "|---|---|---|---:|---:|---:|---:|"]
    for r in between:
        score = f'{r["agreement_among_both_resolved"]:.3f}' if r["agreement_among_both_resolved"] is not None else "N/A"
        lines.append(f'| {r["model_a"]} | {r["model_b"]} | {r["task"]} | {r["common_records"]} | {r["both_resolved"]} | {r["same"]} | {score} |')
    lines += ["", "## Five-voter label distributions", "", "| Model | Task | Resolved absence | Resolved values | Value counts |",
              "|---|---|---:|---:|---|"]
    for r in rows:
        if r["repeat"] != 0 or r["voters"] != 5:
            continue
        absence, values = 0, Counter()
        for key, count in r["label_distribution"].items():
            label = json.loads(key)
            if label["status"] == "not_found":
                absence += count
            else:
                values[str(label["value"])] += count
        lines.append(f'| {r["model"]} | {r["task"]} | {absence} | {sum(values.values())} | {dict(values)} |')
    lines += ["", "## Observed resources", "", "| Model | Peak observed total GPU MiB | Peak summed server RSS MiB |",
              "|---|---:|---:|"]
    for resource_path in sorted(root.glob("*-resources.jsonl")):
        gpu_peak, rss_peak = 0, 0
        for line in resource_path.read_text().splitlines():
            sample = json.loads(line)
            try:
                gpu_peak = max(gpu_peak,float(sample["gpu"].split(",")[2]))
                rss_peak = max(rss_peak,sum(int(row.split()[2]) for row in sample["model_processes"])/1024)
            except (ValueError,IndexError):
                continue
        lines.append(f'| {resource_path.name.removesuffix("-resources.jsonl")} | {gpu_peak:.0f} | {rss_peak:.0f} |')
    lines += ["", "## Interpretation and reproducibility", "",
        "Agreement is not correctness. No model is a gold standard. Explicit null is a valid absence vote; missing fields, invalid JSON, and failed requests are not. Exact raw responses, seeds, durations, and per-field validation are stored in ballots.sqlite. Flat JSON values are represented internally as value/absence ballots for voting, without changing their meaning. No text or label normalization is applied; integer types are strict, duplicate keys and nonfinite numbers are rejected. Markdown-fenced JSON is invalid.", "",
        "The 1/3/5 panels are nested prefixes of the same independent five invocations. They are correlated comparisons, not separately timed workloads. Each invocation sees the same fixed prompt and source text, never previous ballots. The first 200 records are independently repeated with different seeds. Metrics include both-resolved denominators and agreement divided by all common records, so abstention cannot disappear from comparisons. Label distributions reveal absence-dominated agreement.", "",
        "Samples are deterministic physical-block samples from a local Reddit database, ordered by a fixed hash. They are not population-representative or balanced diagnostic sets. Exact records, SQL, exclusions, sizes, and SHA-256 hashes are in sample-manifest.json and the task JSONL files. The appeal sample is restricted to DebateAVegan. Context overflow is a failed request; no silent truncation occurs.", "",
        "The exact user-approved short prompts and JSON schemas are saved in prompts.json. No extra task instructions or system message are added. Age extraction has no current-age qualifier. Appeal values are ethos, pathos, logos, or null. JSON-schema constrained decoding is enabled for both tasks; temperature is 0.7 and output is limited to 192 tokens. Thinking is disabled through chat-template kwargs where supported. Both tasks must pass a structured-output preflight before a model is benchmarked. Malformed or truncated generations remain invalid measured ballots. Only request failures trigger the failure-rate circuit breaker; task duration is also bounded. Partial denominators remain visible. The changed prompts and decoding mean differences from the stopped run cannot be attributed to prompt wording alone.", "",
        "Timing JSON measures warm five-ballot throughput including durable SQLite writes. Per-ballot median and p95 are in metrics.csv. Cold startup includes loading and initialization. Resource JSONL records total GPU usage and process RSS; GPU readings include the display and other activity, RSS sums may double-count shared pages, and two-second sampling can miss peaks. GPU memory reservations are not equivalent to actual weight memory. Model configs, weight bytes, runtime versions, commands, and overrides are in matrix-status.json; newly downloaded models are pinned by revision in acquisition.json.", "",
        "Repeat consistency and every pairwise model comparison are in metrics.json. disagreement-examples.json gives source IDs and observed vote distributions for qualitative inspection without assigning a supposedly correct answer.", "",
        "Reproduce using the Python modules experiments.acquire, experiments.sample, and experiments.matrix, each with --output pointing to a fresh directory outside the source checkout. Use the saved revisions, sample files and settings for exact input reproduction. Sampling against a changed physical database can change the cohort. GPU inference is not guaranteed bitwise deterministic."]
    (root / "REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__ == "__main__":
    import argparse
    from .common import artifact_root
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True)
    generate(artifact_root(p.parse_args().output))
