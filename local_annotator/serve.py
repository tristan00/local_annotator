import subprocess
import sys
import time
from urllib.request import urlopen
from .models import MODELS

PORT = 8090


def serve(name, max_model_len, max_num_seqs, minutes):
    started = time.perf_counter()
    model = MODELS[name]
    command = " ".join(["cd /root && VLLM_USE_FLASHINFER_SAMPLER=0 timeout", str(minutes * 60), "/root/vllm/bin/vllm serve",
                        model["path"], "--served-model-name", name, "--host 0.0.0.0 --port", str(PORT),
                        "--max-model-len", str(max_model_len), "--max-num-seqs", str(max_num_seqs), *model["serve"],
                        "> /root/vllm.log 2>&1"])
    print(f"enter serve {command}", flush=True)
    process = subprocess.Popen(["wsl", "-d", "Ubuntu-24.04", "-u", "root", "--", "bash", "-c", command])
    while process.poll() is None:
        try:
            with urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2):
                print(f"ready {name} seconds={time.perf_counter() - started:.0f}", flush=True)
                break
        except OSError:
            time.sleep(5)
    process.wait()
    print(f"exit serve {name} code={process.returncode} seconds={time.perf_counter() - started:.0f}", flush=True)


if __name__ == "__main__":
    serve(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
