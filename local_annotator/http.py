"""Thread-safe pooled HTTP adapter for local OpenAI-compatible servers."""
import http.client
import json
from queue import LifoQueue, Empty, Full
from urllib.parse import urlsplit
from .core import ModelResponse


class OpenAIAdapter:
    """Reuse connections across voters/records; timeout bounds socket waits.

    Call close() once all in-flight invocations have finished to release idle sockets.
    Transport errors are not retried here: run_annotation owns the retry budget.
    """
    def __init__(self, url, model, temperature=0.7, max_tokens=128):
        self.config = dict(url=url, model=model, temperature=temperature, max_tokens=max_tokens,
                           top_p=0.9, thinking=False, transport='pooled-http-v1')
        self._url = urlsplit(url)
        if self._url.scheme not in ('http', 'https') or not self._url.hostname or self._url.username or self._url.fragment:
            raise ValueError('Expected an HTTP(S) endpoint without embedded credentials or fragment')
        self._pool = LifoQueue(maxsize=64)

    def __call__(self, messages, *, timeout, seed):
        c = self.config
        body = dict(model=c['model'], messages=messages, temperature=c['temperature'],
                    max_tokens=c['max_tokens'], top_p=c['top_p'], seed=seed,
                    chat_template_kwargs={'enable_thinking': False})
        try:
            connection = self._pool.get_nowait()
        except Empty:
            cls = http.client.HTTPSConnection if self._url.scheme == 'https' else http.client.HTTPConnection
            connection = cls(self._url.hostname, self._url.port, timeout=timeout)
        connection.timeout = timeout
        if connection.sock is not None:
            connection.sock.settimeout(timeout)
        path = self._url.path or '/'
        if self._url.query:
            path += '?' + self._url.query
        try:
            connection.request('POST', path, json.dumps(body).encode(), {'Content-Type': 'application/json'})
            response = connection.getresponse()
            raw = response.read()
            if not 200 <= response.status < 300:
                raise RuntimeError(f'HTTP {response.status}: {raw[:1000].decode(errors="replace")}')
            data = json.loads(raw)
        except BaseException:
            connection.close()
            raise
        else:
            try:
                self._pool.put_nowait(connection)
            except Full:
                connection.close()
        choice = data['choices'][0]
        return ModelResponse(choice['message']['content'],
                             {'usage': data.get('usage'), 'finish_reason': choice.get('finish_reason'),
                              'model': data.get('model')})

    def close(self):
        while True:
            try:
                self._pool.get_nowait().close()
            except Empty:
                break

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
