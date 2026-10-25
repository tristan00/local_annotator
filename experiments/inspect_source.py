import json
from pathlib import Path
import psycopg

with psycopg.connect("host=127.0.0.1 port=5499 dbname=reddit user=postgres", connect_timeout=10,
                     options="-c default_transaction_read_only=on -c statement_timeout=10000") as c:
    info = {}
    for key, sql in {
        "version": "SELECT version()",
        "indexes": "SELECT indexname,indexdef FROM pg_indexes WHERE tablename='texts'",
        "estimate": "SELECT reltuples,pg_total_relation_size(oid) FROM pg_class WHERE relname='texts'",
        "activity": "SELECT state,wait_event_type,wait_event,left(query,200) FROM pg_stat_activity WHERE datname='reddit'",
    }.items():
        info[key] = c.execute(sql).fetchall()
    print(json.dumps(info, indent=2))
    Path("measurements/source_inventory.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
