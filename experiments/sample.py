"""Run with python -m experiments.sample --output EXTERNAL_DIRECTORY.

Requires psycopg; connects read-only to the existing local Reddit database.
Fixed block sampling is reproducible against an unchanged physical database,
not a statistically independent simple random sample of the population.
"""
import argparse
import hashlib
import json
from .common import artifact_root, write_json


def main():
    import psycopg
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--dsn", default="host=127.0.0.1 port=5499 dbname=reddit user=postgres")
    args = parser.parse_args()
    root = artifact_root(args.output)
    manifest = {"seed": 20261026, "target": args.count, "gold_labels": False,
                "selection": "Fixed physical block sample, then MD5 ordering; not population representative",
                "tasks": {}}
    with psycopg.connect(args.dsn, connect_timeout=10,
                        options="-c default_transaction_read_only=on -c statement_timeout=180000") as conn:
        for task, percent, condition in [("age", 0.5, "TRUE"), ("appeal", 10, "subreddit='debateavegan'")]:
            path = root / (task + ".jsonl")
            if path.exists():
                raise FileExistsError(f"Refusing to reuse or overwrite an existing sample: {path}")
            sql = f"""SELECT record_id,text,subreddit FROM texts
                TABLESAMPLE SYSTEM ({percent}) REPEATABLE (20261026)
                WHERE {condition} AND record_type='comment' AND text_status='available'
                AND char_length(text) BETWEEN 20 AND 6000
                AND text NOT IN ('[deleted]','[removed]')
                ORDER BY md5('20261026'||record_id),record_id LIMIT %s"""
            rows = conn.execute(sql, (args.count,)).fetchall()
            if not rows:
                raise RuntimeError("No source records for " + task)
            with path.open("w", encoding="utf-8") as stream:
                for rid, text, subreddit in rows:
                    stream.write(json.dumps({"id": rid, "text": text, "subreddit": subreddit}, ensure_ascii=False) + "\n")
            manifest["tasks"][task] = {"n": len(rows), "sql": sql,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "shortfall": max(0, args.count-len(rows)), "subreddits": len(set(r[2] for r in rows))}
            write_json(root / "sample-manifest.json", manifest)
            print(task, len(rows), flush=True)


if __name__ == "__main__":
    main()
