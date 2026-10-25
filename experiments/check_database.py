import sqlite3
from pathlib import Path
try:
    from .paths import ARTIFACT_ROOT, DATA, MEASUREMENTS
except ImportError:
    from paths import ARTIFACT_ROOT, DATA, MEASUREMENTS

p=MEASUREMENTS / 'attempts.sqlite'
print(p,p.exists(),flush=True)
c=sqlite3.connect(p)
print(c.execute('pragma quick_check').fetchall(),flush=True)
print(c.execute('select run_id,count(*) from results group by run_id').fetchall(),flush=True)
c.close()
