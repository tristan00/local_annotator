import json
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from threading import Thread
import unittest
from local_annotator import OpenAIAdapter

class HTTPTests(unittest.TestCase):
    def test_reuses_connection_and_preserves_request_settings(self):
        requests=[]
        class Handler(BaseHTTPRequestHandler):
            protocol_version='HTTP/1.1'
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append((self.client_address,body))
                raw=json.dumps({'choices':[{'message':{'content':'{"age":25}'},'finish_reason':'stop'}]}).encode()
                self.send_response(200);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
            def log_message(self,*args):pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with OpenAIAdapter(f'http://127.0.0.1:{server.server_port}/v1/chat/completions','test') as adapter:
                for seed in (1,2,3):
                    response=adapter([{'role':'user','content':'text'}],timeout=2,seed=seed)
                    self.assertEqual(response.content,'{"age":25}')
            self.assertEqual(len({r[0] for r in requests}),1)
            self.assertEqual([r[1]['seed'] for r in requests],[1,2,3])
        finally:
            server.shutdown();server.server_close();thread.join()

if __name__=='__main__':unittest.main()
