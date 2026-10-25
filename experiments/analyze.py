"""Recompute descriptive metrics from preserved ballots; never infer correctness."""
import itertools
import json
from collections import Counter
from pathlib import Path
import sqlite3
import statistics


def vote(ballots, field, n):
    values = [b['values'][field] for b in ballots[:n] if field in b['values']]
    counts = Counter(json.dumps(v) for v in values)
    winner = next((v for v, c in counts.items() if c > n / 2), None)
    return winner, values


def ratio(a, b):
    return a / b if b else None


def main():
    root = Path('measurements')
    db = sqlite3.connect(root / 'attempts.sqlite')
    sessions = [json.loads(s) for s in (root / 'sessions.jsonl').read_text().splitlines()] if (root / 'sessions.jsonl').exists() else []
    metrics, outputs, examples = [], {}, []
    for run_id, raw in db.execute('SELECT run_id,config FROM runs ORDER BY run_id'):
        config = json.loads(raw)
        rows = [json.loads(r[0]) for r in db.execute('SELECT payload FROM results WHERE run_id=?', (run_id,))]
        task = config['task']
        if not rows:
            continue
        times = [a['seconds'] for r in rows for b in r['ballots'] for a in b['attempts']]
        statuses = Counter(a['status'] for r in rows for b in r['ballots'] for a in b['attempts'])
        elapsed = sum(s['seconds'] for s in sessions if s['run_id'] == run_id)
        for n in (1, 3, 5):
            if n > config['voters']:
                continue
            resolved, unanimous, pairs, equal, valid = 0, 0, 0, 0, 0
            distribution = Counter()
            decisions = {}
            example_count = 0
            for r in rows:
                winner, values = vote(r['ballots'], task, n)
                decisions[r['id']] = winner
                valid += len(values)
                resolved += winner is not None
                unanimous += len(values) == n and len(set(map(json.dumps, values))) == 1
                for a, b in itertools.combinations(values, 2):
                    pairs += 1
                    equal += a == b
                if winner is not None:
                    distribution[json.loads(winner)] += 1
                if n == 5 and example_count < 3 and config['settings'].get('transport') == 'pooled-http-v1' and len(set(map(json.dumps, values))) > 1:
                    examples.append(dict(run_id=run_id, id=r['id'], ballots=values, voted=winner))
                    example_count += 1
            key = (config['model'], task, config['repeat'], n, config['label'])
            if config['settings'].get('transport') == 'pooled-http-v1':
                outputs[key] = decisions
            metrics.append(dict(run_id=run_id, model=config['model'], task=task, repeat=config['repeat'], voters=n,
                transport=config['settings'].get('transport', 'unpooled-urllib'),
                records=len(rows), planned_records=config['limit'], resolved=resolved, unresolved=len(rows)-resolved,
                valid_ballots=valid, scheduled_ballots=len(rows)*n, unanimous=unanimous,
                equal_valid_pairs=equal, valid_pairs=pairs, pair_agreement=ratio(equal,pairs),
                unanimous_rate=ratio(unanimous,len(rows)), unresolved_rate=ratio(len(rows)-resolved,len(rows)),
                value_distribution=dict(distribution), attempt_statuses=statuses,
                request_latency_p50=statistics.median(times), request_latency_p95=sorted(times)[min(len(times)-1,int(len(times)*.95))],
                measured_voters=config['voters'], completed_session_seconds=elapsed,
                measured_records_per_second=ratio(sum(s['completed_this_session'] for s in sessions if s['run_id']==run_id),elapsed)))
    comparisons=[]
    for (ka, a), (kb, b) in itertools.combinations(outputs.items(), 2):
        if ka[1] != kb[1] or ka[3] != kb[3]:
            continue
        if ka[4] == 'pilot' or kb[4] == 'pilot':
            continue
        # Prefer full runs over the preliminary 1,000-record warmup.
        if any(k[0] == ka[0] and k[1:4] == ka[1:4] and k[4] == 'full' for k in outputs) and ka[4] == 'main':
            continue
        if any(k[0] == kb[0] and k[1:4] == kb[1:4] and k[4] == 'full' for k in outputs) and kb[4] == 'main':
            continue
        kind = 'between_model' if ka[0] != kb[0] and ka[2] == kb[2] == 0 else 'repeat' if ka[0] == kb[0] and ka[2] != kb[2] else None
        if kind is None:
            continue
        common=set(a)&set(b)
        both=[rid for rid in common if a[rid] is not None and b[rid] is not None]
        matches=sum(a[rid]==b[rid] for rid in both)
        comparisons.append(dict(kind=kind,a=ka,b=kb,common_records=len(common),both_resolved=len(both),equal=matches,agreement=ratio(matches,len(both))))
    (root/'metrics.json').write_text(json.dumps(dict(runs=metrics,comparisons=comparisons,disagreement_examples=examples),indent=2))
    lines=['# Local Annotator benchmark report','',
        'These are measurements of agreement and execution, not accuracy. No gold labels or model-derived ground truth are used.', '',
        '## Quantitative results','',
        'Voter counts 1/3/5 are nested prefixes of the same five-ballot runs. Pair agreement excludes invalid/missing ballots; unanimity and unresolved rates use all completed records. A one-voter result has no pair-agreement denominator. Timings describe the executed five-voter configuration only.', '',
        '| Model | Task | Repeat | Voters | Records | Unresolved | Valid ballots | Equal/valid pairs | Unanimous |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for m in metrics:
        if m['transport'] != 'pooled-http-v1':
            continue
        lines.append(f"| {m['model']} | {m['task']} | {m['repeat']} | {m['voters']} | {m['records']} | {m['unresolved']} | {m['valid_ballots']}/{m['scheduled_ballots']} | {m['equal_valid_pairs']}/{m['valid_pairs']} | {m['unanimous']} |")
    lines += ['', 'The main table and cross-run comparisons use the corrected pooled HTTP adapter. The initial pilot, warmup and interrupted unpooled run remain in the timing table and metrics JSON as diagnostic evidence. The unpooled Windows run exhausted ephemeral socket ports; its unresolved outcomes are not model-quality measurements.']
    lines += ['', '### Actual execution timing','', '| Run | Valid / invalid / failed attempts | Session seconds | Records/s | Request p50 / p95 seconds |','|---|---:|---:|---:|---:|']
    for m in metrics:
        if m['voters'] != m['measured_voters']:
            continue
        s=m['attempt_statuses']; rate=m['measured_records_per_second']
        lines.append(f"| {m['run_id']} | {s.get('valid',0)} / {s.get('invalid',0)} / {s.get('failed',0)} | {m['completed_session_seconds']:.2f} | {rate:.2f} | {m['request_latency_p50']:.3f} / {m['request_latency_p95']:.3f} |" if rate is not None else f"| {m['run_id']} | {dict(s)} | incomplete | unavailable | unavailable |")
    lines += ['', '### Repeated-run and between-model agreement','', '| Comparison | Task | Voters | Common records | Both resolved | Equal | Agreement |','|---|---|---:|---:|---:|---:|---:|']
    for c in comparisons:
        a,b=c['a'],c['b']; pct=f"{100*c['agreement']:.2f}%" if c['agreement'] is not None else 'N/A'
        lines.append(f"| {c['kind']}: {a[0]} r{a[2]} vs {b[0]} r{b[2]} | {a[1]} | {a[3]} | {c['common_records']} | {c['both_resolved']} | {c['equal']} | {pct} |")
    lines += ['', '## Runtime and resource observations','', '| Model | Weight GiB | Startup seconds | GPU MiB min/max | WSL used RAM GiB min/max | Server process RSS GiB max |','|---|---:|---:|---:|---:|---:|']
    for p in sorted(root.glob('*-runtime.json')):
        r=json.loads(p.read_text()); name=p.name.removesuffix('-runtime.json'); resource=root/(name+'-resources.jsonl')
        samples=[json.loads(s) for s in resource.read_text().splitlines()] if resource.exists() else []
        gpu=[float(s['gpu'].split(',')[0]) for s in samples if s.get('gpu')]
        ram=[(int(s['meminfo']['MemTotal'].split()[0])-int(s['meminfo']['MemAvailable'].split()[0]))/2**20 for s in samples]
        rss=[sum(int(line.split()[2]) for line in s['processes'])/2**20 for s in samples]
        lines.append(f"| {name} | {r['weights_bytes']/2**30:.2f} | {r.get('ready_seconds','not ready')} | {f'{min(gpu):.0f}/{max(gpu):.0f}' if gpu else 'unavailable'} | {f'{min(ram):.2f}/{max(ram):.2f}' if ram else 'unavailable'} | {f'{max(rss):.2f}' if rss else 'unavailable'} |")
    lines += ['', 'Hardware: NVIDIA RTX 5090, 32,607 MiB reported VRAM, Windows host with WSL Linux serving. Exact model configs, quantization, runtime versions, startup commands, WSL RAM snapshots and process RSS are preserved in `measurements/*-runtime.json` and `*-resources.jsonl`. GPU usage includes display/runtime allocation and is not isolated model weight use; sampled peaks may miss short spikes. Startup includes loading/compilation/warmup and is separate from inference sessions. Prefix caches can warm across tasks/repeats. Socket timeouts are not hard total response deadlines.', '',
        '## Sampling and interpretation','',
        'Inputs are archived in data/*.jsonl with SQL and SHA256 fingerprints in data/manifest.json. The fixed 10,000-record target per task is a sample pool; table rows above state actual completed coverage. Age uses comments across subreddits; appeal uses debateavegan. Both use deterministic PostgreSQL block sampling followed by MD5 ordering, so they are not simple random Reddit samples. Physical table changes can alter resampling. Only comment text is supplied, without parents. Task/rubric version, field definitions and seeds are stored in every run config.', '',
        'Exact typed equality is used; no aliases, fuzzy matching, extraction of JSON from prose, or repair of malformed outputs. Explicit not_found is a valid answer; failures and invalid answers are not absence. Five-voter strict majority is three; three-voter strict majority is two. Repeats change seeds. Models share prompts and training influences, so matching ballots do not imply independent evidence.', '',
        '## Qualitative observations','',
        'See measurements/metrics.json for value distributions and stable IDs of observed ballot disagreements. Large not_found shares can dominate age agreement: high agreement alone does not demonstrate successful age extraction. Rhetorical labels reflect a fixed operational rubric, not uniquely true categories. Differences between logos, pathos and mixed warrant inspection of the original comment; this report does not assume any model is correct.', '',
        'All raw attempts and completed results are in measurements/attempts.sqlite. Completed-session timings are in sessions.jsonl; interrupted startup logs remain evidence of failures, not completed runs. Reproduce with the commands in experiments/README.md and regenerate this report using python experiments/analyze.py.']
    Path('REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__ == '__main__':
    main()
