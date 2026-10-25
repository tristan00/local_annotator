"""WSL matrix runner; owns only the server supervisors it starts."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen

p=argparse.ArgumentParser()
p.add_argument('--models',nargs='+',default=['qwen3-1.7b','qwen25-1.5b','smollm2-1.7b','qwen36-nvfp4','qwen38-27b-nvfp4'])
p.add_argument('--limit',type=int,default=10000)
a=p.parse_args()
out=Path('measurements')
for model in a.models:
    stop=out/(model+'.stop')
    stop.unlink(missing_ok=True)
    started=time.time()
    with (out/(model+'-supervisor.log')).open('w') as log:
        server=subprocess.Popen([sys.executable,'experiments/serve.py',model],stdout=log,stderr=subprocess.STDOUT)
        try:
            while time.time()-started<660:
                if server.poll() is not None:
                    raise RuntimeError('server startup failed; see logs')
                try:
                    with urlopen('http://127.0.0.1:8091/v1/models',timeout=2) as r:
                        assert json.load(r)['data'][0]['id']==model
                    break
                except Exception:
                    time.sleep(2)
            else:
                raise TimeoutError('server startup timeout')
            limit=min(a.limit,1000) if 'nvfp4' in model else a.limit
            for repeat,size,label in [(0,limit,'full-v2' if limit==10000 else 'main-v2'),(1,200,'repeat-v2')]:
                subprocess.run([sys.executable,'experiments/benchmark.py',model,'--limit',str(size),'--voters','5','--repeat',str(repeat),'--label',label],check=True,timeout=1800)
        except Exception as e:
            with (out/'matrix_failures.jsonl').open('a') as f:
                f.write(json.dumps({'model':model,'error':str(e),'epoch':time.time()})+'\n')
            print(model,type(e).__name__,str(e),flush=True)
        finally:
            stop.touch()
            try:
                server.wait(timeout=45)
            except subprocess.TimeoutExpired:
                server.terminate()
                server.wait(timeout=10)
            time.sleep(3)
    subprocess.run([sys.executable,'experiments/analyze.py'],check=True)
