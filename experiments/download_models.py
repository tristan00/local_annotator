"""Acquire four small public models, with a 13 GB total file-size guard."""
import json
from pathlib import Path
try:
    from .paths import ARTIFACT_ROOT, DATA, MEASUREMENTS
except ImportError:
    from paths import ARTIFACT_ROOT, DATA, MEASUREMENTS

from huggingface_hub import HfApi, snapshot_download

models = {"qwen3-0.6b": "Qwen/Qwen3-0.6B", "qwen3-1.7b": "Qwen/Qwen3-1.7B",
          "qwen25-1.5b": "Qwen/Qwen2.5-1.5B-Instruct", "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct"}
patterns = ["*.json", "*.safetensors", "tokenizer.model", "*.tiktoken", "merges.txt", "vocab.txt", "*.jinja", "README.md"]
MEASUREMENTS.mkdir(parents=True, exist_ok=True)
inventory = {}
api = HfApi()
import fnmatch
for name, repo in models.items():
    info = api.model_info(repo, files_metadata=True)
    sizes = {s.rfilename: s.size or 0 for s in info.siblings if any(fnmatch.fnmatch(s.rfilename, p) for p in patterns)}
    inventory[name] = {"repo": repo, "revision": info.sha, "bytes": sum(sizes.values()), "files": sizes}
if sum(x["bytes"] for x in inventory.values()) > 13_000_000_000:
    raise RuntimeError("Download guard exceeded; no files downloaded")
(MEASUREMENTS / "download_inventory.json").write_text(json.dumps(inventory, indent=2))
for name, entry in inventory.items():
    print("downloading", name, entry["bytes"], flush=True)
    snapshot_download(entry["repo"], revision=entry["revision"], allow_patterns=patterns,
                      local_dir="/root/models/" + name, max_workers=4)
    print("complete", name, flush=True)
