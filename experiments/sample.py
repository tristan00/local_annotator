"""Read-only PostgreSQL sampling. Run using an environment with psycopg installed."""
import hashlib
import json
from pathlib import Path
import time
import psycopg

out = Path("data")
out.mkdir(exist_ok=True)
started = time.time()
manifest = {"seed": 20261025, "target_per_task": 10000, "source": "local Reddit PostgreSQL texts",
            "exclusions": "non-comments; unavailable/deleted/removed text; fewer than 20 or more than 10000 characters",
            "selection": "age: SYSTEM 0.5%; appeal: SYSTEM 10%; both REPEATABLE(20261025), then MD5(seed || record_id)",
            "gold_labels": False}
with psycopg.connect("host=127.0.0.1 port=5499 dbname=reddit user=postgres", connect_timeout=10,
                     options="-c default_transaction_read_only=on -c statement_timeout=180000") as conn:
    print("connected", flush=True)
    manifest["subreddit_probe"] = conn.execute("SELECT subreddit,record_type,text_status FROM texts WHERE subreddit='debateavegan' LIMIT 5").fetchall()
    print(manifest["subreddit_probe"], flush=True)
    for task, table, condition in [("age", "texts TABLESAMPLE SYSTEM (0.5) REPEATABLE (20261025)", "TRUE"),
                                   ("appeal", "texts TABLESAMPLE SYSTEM (10) REPEATABLE (20261025)", "subreddit='debateavegan'")]:
        sql = f"""SELECT record_id, text, subreddit, created_utc FROM {table}
                  WHERE {condition} AND record_type='comment' AND text_status='available'
                    AND char_length(text) BETWEEN 20 AND 10000 AND text NOT IN ('[deleted]','[removed]')
                  ORDER BY md5('20261025' || record_id),record_id LIMIT 10000"""
        path = out / (task + ".jsonl")
        if task == "age" and path.exists():
            saved = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            rows = [(r["id"],r["text"],r["subreddit"],r["created_utc"]) for r in saved]
        else:
            rows = conn.execute(sql).fetchall()
        if not rows:
            raise RuntimeError("No records: verify record_type and filters")
        with path.open("w", encoding="utf-8") as stream:
            for rid, text, subreddit, created in rows:
                stream.write(json.dumps(dict(id=rid, text=text, subreddit=subreddit, created_utc=created), ensure_ascii=False) + "\n")
        manifest[task] = {"n": len(rows), "sql": sql, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "subreddits": len(set(r[2] for r in rows)), "characters": sum(len(r[1]) for r in rows),
                          "target_shortfall": max(0, 10000 - len(rows))}
        print(task, manifest[task], flush=True)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
manifest["seconds"] = time.time() - started
(out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
