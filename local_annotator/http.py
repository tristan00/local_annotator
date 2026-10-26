"""Thread-local keep-alive HTTP adapter for local OpenAI-compatible servers."""
import http.client
import json
import threading
from urllib.parse import urlsplit


class LocalChat:
    def __init__(self, endpoint, model, sampling, *, max_tokens, timeout, schema=None, regex=None, messages=()):
        url = urlsplit(endpoint)
        if url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("LocalChat requires a loopback HTTP endpoint")
        self.host, self.port = url.hostname, url.port or 80
        self.path = url.path.rstrip("/") + "/chat/completions"
        self.model, self.sampling, self.max_tokens = model, dict(sampling), max_tokens
        self.timeout = timeout
        self.local = threading.local()
        self.schema = schema
        self.regex = regex
        self.messages = [dict(message) for message in messages]

    def __call__(self, prompt, seed):
        if not getattr(self.local, "connection", None):
            self.local.connection = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        connection = self.local.connection
        payload = {"model": self.model, "messages": [*self.messages, {"role": "user", "content": prompt}],
                   **self.sampling, "max_tokens": self.max_tokens, "seed": seed,
                   "chat_template_kwargs": {"enable_thinking": False}}
        if self.schema is not None:
            payload["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "annotation", "strict": True, "schema": self.schema}}
        if self.regex is not None:
            payload["structured_outputs"] = {"regex": self.regex}
        try:
            connection.request("POST", self.path, json.dumps(payload), {"Content-Type": "application/json"})
            response = connection.getresponse()
            body = response.read()
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}: {body.decode(errors='replace')[:1500]}")
            data = json.loads(body)
            choice = data["choices"][0]
            if choice.get("finish_reason") != "stop":
                error = ValueError(f"Incomplete generation: {choice.get('finish_reason')}")
                error.raw = choice["message"].get("content")
                raise error
            return choice["message"]["content"]
        except Exception:
            connection.close()
            self.local.connection = None
            raise

    def close(self):
        if getattr(self.local, "connection", None):
            self.local.connection.close()
            self.local.connection = None
