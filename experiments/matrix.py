"""Bounded local benchmark orchestration. Run inside WSL's vLLM environment.

python -m experiments.matrix --output /mnt/d/local_annotator_runs/RUN
Owns only servers it starts; never kills an existing service. Acquisition and
sampling must finish first. Artifacts and model weights stay outside checkout.
"""
import argparse
from dataclasses import asdict
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
from urllib.request import urlopen
from local_annotator import Record, run_annotation
from local_annotator.http import LocalChat
from .benchmark import run
from .common import artifact_root, write_json
from .tasks import TASKS, PROMPTS, SCHEMAS, VERSION, prompt_builder

MODELS = ["qwen3-0.6b", "smollm2-1.7b", "phi4-mini", "granite33-2b", "olmo2-1b",
          "mistral7b", "qwen25-1.5b", "qwen3-1.7b", "qwen36-nvfp4", "qwen38-27b-nvfp4"]


def monitor(root, name, stop):
    with (root / (name + "-resources.jsonl")).open("w") as stream:
        while not stop.is_set():
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
            processes = subprocess.run(["ps", "-eo", "pid,ppid,rss,args"], capture_output=True, text=True, timeout=10)
            stream.write(json.dumps({"epoch": time.time(), "gpu": gpu.stdout.strip(),
                "system_memory": Path("/proc/meminfo").read_text(),
                "model_processes": [line for line in processes.stdout.splitlines() if "VLLM" in line or "vllm serve" in line]}) + "\n")
            stream.flush()
            stop.wait(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--cache", default="/root/models")
    args = parser.parse_args()
    root = artifact_root(args.output)
    if (root / "matrix-status.json").exists():
        raise FileExistsError("Use a fresh output directory; refusing to overwrite a previous matrix")
    write_json(root / "planned-models.json", args.models)
    write_json(root / "prompts.json", {"version": VERSION, "templates": PROMPTS, "schemas": SCHEMAS,
                                      "messages": "one user message; no system message", "output_format": "flat"})
    source = Path(__file__).resolve().parents[1]
    write_json(root / "source-hashes.json", {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
               for directory in (source / "local_annotator", source / "experiments") for p in directory.glob("*.py")})
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 8091))  # An occupied port is not ours to clear.
    status = {}
    for name in args.models:
        path = Path(args.cache) / name
        meta = {"model": name, "started": time.time(), "status": "checking"}
        status[name] = meta
        proc, log, thread = None, None, None
        stop = threading.Event()
        try:
            config = json.loads((path / "config.json").read_text())
            index = path / "model.safetensors.index.json"
            weights = set(json.loads(index.read_text())["weight_map"].values()) if index.exists() else {"model.safetensors"}
            missing = [w for w in weights if not (path / w).is_file()]
            if missing:
                raise RuntimeError("Missing weight files: " + repr(missing))
            meta.update(config=config, weights_bytes=sum((path/w).stat().st_size for w in weights),
                        versions={n:importlib.metadata.version(n) for n in ("vllm", "torch", "transformers")})
            command = ["/root/vllm/bin/vllm", "serve", str(path), "--host", "127.0.0.1", "--port", "8091",
                       "--served-model-name", name, "--max-model-len", "4096", "--max-num-seqs", "32",
                       "--gpu-memory-utilization", "0.85", "--language-model-only", "--enforce-eager"]
            env = os.environ.copy()
            # Use the installed, matching compiler/header tree; do not modify it.
            toolkit = Path("/root/vllm/lib/python3.12/site-packages/nvidia/cu13")
            env["CUDA_HOME"] = str(toolkit)
            env["PATH"] = str(toolkit / "bin") + ":/root/vllm/bin:" + env.get("PATH", "")
            env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            meta.update(command=command, environment={k:env[k] for k in ("CUDA_HOME", "PATH", "VLLM_USE_FLASHINFER_SAMPLER")},status="loading")
            write_json(root / "matrix-status.json", status)
            log = (root / (name + "-server.log")).open("w")
            proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            thread = threading.Thread(target=monitor, args=(root,name,stop), daemon=True)
            thread.start()
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f"Server exited with {proc.returncode}; see server log")
                try:
                    with urlopen("http://127.0.0.1:8091/health", timeout=2):
                        break
                except OSError:
                    time.sleep(2)
            else:
                raise TimeoutError("Server startup exceeded 10 minutes")
            meta["cold_start_seconds"] = time.time() - meta["started"]
            for task, text in (("age", "I am 25 years old."),
                               ("appeal", "The proposal costs less, so it saves money.")):
                adapter = LocalChat("http://127.0.0.1:8091/v1", name, {"temperature": 0.7}, max_tokens=192, timeout=60, schema=SCHEMAS[task])
                try:
                    preflight = next(run_annotation(None, lambda _: [Record("synthetic-preflight", text)],
                        TASKS[task], adapter, annotators=3, seed=42,
                        prompt_builder=prompt_builder(task), output_format="flat"))
                finally:
                    adapter.close()
                write_json(root / (name + "-" + task + "-preflight.json"), asdict(preflight))
                if not all(task in b.answers and b.error is None for b in preflight.ballots):
                    raise RuntimeError(f"Structured-output preflight failed for {task}; see saved ballots")
            meta["status"] = "benchmarking"
            write_json(root / "matrix-status.json", status)
            run(root, name)
            meta["status"] = "completed"
        except Exception as exc:
            meta.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            print(name, meta["error"], flush=True)
        except BaseException:
            meta.update(status="interrupted")
            raise
        finally:
            if proc is not None:
                # Always clean up the owned group, even if the parent exited.
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=10)
            stop.set()
            if thread:
                thread.join(timeout=15)
            if log:
                log.close()
            meta["ended"] = time.time()
            write_json(root / "matrix-status.json", status)
            print(name, meta["status"], flush=True)
    from .report import generate
    generate(root)


if __name__ == "__main__":
    main()
