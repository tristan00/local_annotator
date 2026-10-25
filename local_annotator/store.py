"""Optional local SQLite persistence; never writes to the caller's source."""
import json
import sqlite3
import threading
import time
import uuid


class AttemptStore:
    def __init__(self, path):
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        self.connection.executescript('''
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS runs (
              run_id TEXT PRIMARY KEY, created REAL NOT NULL, config TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS attempts (
              attempt_id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, record_id TEXT NOT NULL,
              voter INTEGER NOT NULL, attempt INTEGER NOT NULL, payload TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS attempts_run ON attempts(run_id, record_id);
            CREATE TABLE IF NOT EXISTS results (
              run_id TEXT NOT NULL, record_id TEXT NOT NULL, payload TEXT NOT NULL,
              PRIMARY KEY(run_id, record_id));
        ''')

    def start_run(self, config, run_id=None):
        run_id = run_id or str(uuid.uuid4())
        with self.lock, self.connection:
            self.connection.execute("INSERT INTO runs VALUES (?,?,?)", (run_id, time.time(), json.dumps(config)))
        return run_id

    def attempt_sink(self, run_id):
        def save(event):
            with self.lock, self.connection:
                self.connection.execute("INSERT INTO attempts(run_id,record_id,voter,attempt,payload) VALUES (?,?,?,?,?)",
                    (run_id, json.dumps(event["record_id"]), event["voter"], event["attempt"], json.dumps(event)))
        return save

    def save_result(self, run_id, result):
        with self.lock, self.connection:
            self.connection.execute("INSERT INTO results VALUES (?,?,?)",
                                    (run_id, json.dumps(result["id"]), json.dumps(result)))

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
