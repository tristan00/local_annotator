import json
import os
from contextlib import closing
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
from threading import Thread
import unittest

class BenchmarkTests(unittest.TestCase):
    def test_both_tasks_resume_without_duplicate_attempts(self):
        class Handler(BaseHTTPRequestHandler):
            protocol_version='HTTP/1.1'
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                raw=json.dumps({'choices':[{'message':{'content':'{"age":25,"appeal":"logos"}'},'finish_reason':'stop'}]}).encode()
                self.send_response(200);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
            def log_message(self,*args):pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);(root/'data').mkdir()
                for task in ('age','appeal'):
                    (root/'data'/f'{task}.jsonl').write_text('\n'.join(json.dumps({'id':str(i),'text':'I am 25.'}) for i in range(2)))
                script=Path(__file__).resolve().parents[1]/'experiments'/'benchmark.py'
                args=[sys.executable,str(script),'test','--url',f'http://127.0.0.1:{server.server_port}/v1/chat/completions','--limit','2','--workers','2']
                for _ in range(2):
                    run=subprocess.run(args,cwd=tmp,env={**os.environ, "LOCAL_ANNOTATOR_ARTIFACTS":tmp},capture_output=True,text=True,timeout=20)
                    self.assertEqual(run.returncode,0,run.stderr)
                with closing(sqlite3.connect(root/'measurements'/'attempts.sqlite')) as db:
                    self.assertEqual(db.execute('select count(*) from results').fetchone()[0],4)
                    self.assertEqual(db.execute('select count(*) from attempts').fetchone()[0],20)
                self.assertIn('todo=0',run.stdout)
        finally:
            server.shutdown();server.server_close();thread.join()

if __name__=='__main__':unittest.main()
