from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import unittest
from local_annotator.http import LocalChat


class AdapterTests(unittest.TestCase):
    def test_connection_reuse_and_failure_recovery(self):
        peers, requests = [], []
        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            def do_POST(self):
                peers.append(self.client_address)
                requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                body = json.dumps({"choices":[{"finish_reason":"stop", "message":{"content":"{}"}}]}).encode()
                self.send_response(503 if len(peers)==3 else 200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def log_message(self,*args):
                pass
        server = ThreadingHTTPServer(("127.0.0.1",0),Handler)
        thread = threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        schema={"type":"object","properties":{"age":{"type":["integer","null"]}},"required":["age"],"additionalProperties":False}
        adapter=LocalChat(f"http://127.0.0.1:{server.server_port}/v1","test",{"temperature":0.7},max_tokens=192,timeout=60,schema=schema)
        try:
            self.assertEqual(adapter("one",1),"{}")
            self.assertEqual(adapter("two",2),"{}")
            self.assertEqual(peers[0],peers[1])
            with self.assertRaisesRegex(RuntimeError,"HTTP 503"):
                adapter("three",3)
            self.assertEqual(adapter("four",4),"{}")
            self.assertNotEqual(peers[2],peers[3])
            self.assertEqual(len(requests[1]["messages"]),1)
            self.assertEqual(requests[1]["messages"][0]["content"],"two")
            self.assertEqual(requests[1]["response_format"]["type"],"json_schema")
            self.assertEqual(requests[1]["response_format"]["json_schema"]["schema"],schema)
            prefixed = LocalChat(f"http://127.0.0.1:{server.server_port}/v1", "test", {"temperature": 0}, max_tokens=8, timeout=60,
                                 messages=[{"role": "system", "content": "rules"}, {"role": "user", "content": "example"},
                                           {"role": "assistant", "content": "none"}])
            self.assertEqual(prefixed("five", 5), "{}")
            self.assertEqual([m["role"] for m in requests[-1]["messages"]], ["system", "user", "assistant", "user"])
            self.assertEqual(requests[-1]["messages"][-1]["content"], "five")
            prefixed.close()
        finally:
            adapter.close()
            server.shutdown()
            server.server_close()
            thread.join()
