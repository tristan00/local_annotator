import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

from .models import MODELS

DISTRO = "Ubuntu-24.04"
ROOT = "/root/models"
SOURCE = Path(__file__).resolve().with_name("vllm_serve.sh")
SCRIPT = f"/mnt/{SOURCE.drive[0].lower()}{SOURCE.as_posix()[2:]}"


def wsl(command, **kwargs):
    return subprocess.run(["wsl", "-d", DISTRO, "-u", "root", "--", "bash", "-c", command], stdin=subprocess.DEVNULL,
                          capture_output=True, encoding="utf-8", errors="replace", **kwargs)


def pattern(name):
    return f"vllm serve {ROOT}/[{name[0]}]{name[1:]} "


def start(name, port, gpu_memory_utilization, max_num_seqs=256, max_model_len=8192, minutes=600, startup_seconds=900):
    started = time.perf_counter()
    log = f"{ROOT}/serve_{name}.log"
    arguments = ["--served-model-name", name, "--host", "127.0.0.1", "--port", str(port), "--max-model-len", str(max_model_len),
                 "--max-num-seqs", str(max_num_seqs), "--gpu-memory-utilization", str(gpu_memory_utilization), "--enable-prefix-caching",
                 *MODELS[name]["serve"]]
    command = f"cd /root && timeout {minutes * 60} bash {SCRIPT} {ROOT}/{name} {' '.join(arguments)} > {log} 2>&1"
    print(f"enter start {name} port={port} command={command}", flush=True)
    process = subprocess.Popen(["wsl", "-d", DISTRO, "-u", "root", "--", "bash", "-c", command], stdin=subprocess.DEVNULL)
    while time.perf_counter() - started < startup_seconds:
        if process.poll() is not None:
            raise RuntimeError(f"{name} server exited with {process.returncode}:\n{wsl(f'tail -40 {log}').stdout}")
        try:
            with urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                print(f"exit start {name} ready seconds={time.perf_counter() - started:.0f}", flush=True)
                return process
        except OSError:
            time.sleep(5)
    stop(name)
    raise TimeoutError(f"{name} server not ready after {startup_seconds}s:\n{wsl(f'tail -40 {log}').stdout}")


def stop(*names):
    started = time.perf_counter()
    for name in names:
        wsl(f"pkill -f '{pattern(name)}'")
    while any(wsl(f"pgrep -f '{pattern(name)}'").returncode == 0 for name in names):
        time.sleep(1)
    print(f"exit stop {' '.join(names)} seconds={time.perf_counter() - started:.0f}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "stop":
        stop(*sys.argv[2:])
    else:
        start(sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), *map(int, sys.argv[5:]))
        print("serving; stop with: python -m local_annotator.serve stop " + sys.argv[2], flush=True)
