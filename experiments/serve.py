"""Run in WSL. Own and terminate only the process group created here."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from urllib.request import urlopen

p = argparse.ArgumentParser()
p.add_argument("model", choices=["qwen36-nvfp4", "qwen38-27b-nvfp4", "qwen3-0.6b", "qwen3-1.7b", "qwen25-1.5b", "smollm2-1.7b"])
p.add_argument("--seconds", type=int, default=10800)
a = p.parse_args()
out = Path("measurements")
out.mkdir(exist_ok=True)
model = Path("/root/models") / a.model
index_path = model / "model.safetensors.index.json"
shards = set(json.loads(index_path.read_text())["weight_map"].values()) if index_path.exists() else {"model.safetensors"}
missing = [s for s in shards if not (model / s).is_file()]
if missing:
    raise RuntimeError(missing)
cmd = ["/root/vllm/bin/vllm", "serve", str(model), "--host", "127.0.0.1", "--port", "8091",
       "--served-model-name", a.model, "--max-model-len", "4096", "--max-num-seqs", "64",
       "--gpu-memory-utilization", "0.90", "--language-model-only"]
started = time.time()
log = (out / (a.model + "-server.log")).open("w")
env = os.environ.copy()
env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
toolkit = "/root/local-annotator-cuda13/nvidia/cu13"
env["PATH"] = "/root/vllm/bin:" + toolkit + "/bin:" + env.get("PATH", "")
env["CUDA_HOME"] = toolkit
stop = out / (a.model + ".stop")
if stop.exists():
    raise RuntimeError("Remove this model's old stop marker before a new run")
proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
meta = {"command": cmd, "pid": proc.pid, "started_epoch": started,
        "weights_bytes": sum((model / s).stat().st_size for s in shards), "environment_overrides": {k: env[k] for k in ("VLLM_USE_FLASHINFER_SAMPLER", "PATH", "CUDA_HOME")},
        "versions": {n: importlib.metadata.version(n) for n in ("vllm", "transformers", "torch")},
        "config": json.loads((model / "config.json").read_text())}
stop = out / (a.model + ".stop")
if stop.exists():
    raise RuntimeError("Remove this model's old stop marker before a new run")
try:
    while time.time() - started < 600:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited {proc.returncode}")
        try:
            with urlopen("http://127.0.0.1:8091/health", timeout=2):
                break
        except Exception:
            time.sleep(2)
    else:
        raise TimeoutError("startup exceeded 600 seconds")
    meta["ready_seconds"] = time.time() - started
    print(json.dumps({"model": a.model, "ready_seconds": meta["ready_seconds"]}), flush=True)
    (out / (a.model + "-runtime.json")).write_text(json.dumps(meta, indent=2))
    with (out / (a.model + "-resources.jsonl")).open("w") as stream:
        while time.time() - started < a.seconds and proc.poll() is None and not stop.exists():
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
            mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
            processes = subprocess.run(["ps", "-eo", "pid,ppid,rss,args"], capture_output=True, text=True).stdout
            stream.write(json.dumps({"epoch": time.time(), "gpu": gpu, "meminfo": mem,
                                    "processes": [line for line in processes.splitlines() if "VLLM" in line or str(proc.pid) in line]}) + "\n")
            stream.flush()
            time.sleep(5)
finally:
    meta["ended_epoch"] = time.time()
    (out / (a.model + "-runtime.json")).write_text(json.dumps(meta, indent=2))
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
    log.close()
