import sqlite3
from pathlib import Path
p=Path('measurements/attempts.sqlite').resolve()
print(p,p.exists(),flush=True)
c=sqlite3.connect(p)
print(c.execute('pragma quick_check').fetchall(),flush=True)
print(c.execute('select run_id,count(*) from results group by run_id').fetchall(),flush=True)
c.close()
