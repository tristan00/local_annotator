"""Acquire a bounded, pinned set of public models into an external model cache."""
import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
from .common import artifact_root, write_json

CATALOG = {
    "phi4-mini": "microsoft/Phi-4-mini-instruct",
    "granite33-2b": "ibm-granite/granite-3.3-2b-instruct",
    "olmo2-1b": "allenai/OLMo-2-0425-1B-Instruct",
    "mistral7b": "mistralai/Mistral-7B-Instruct-v0.3",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", default="/root/models")
    args = parser.parse_args()
    root, cache = artifact_root(args.output), artifact_root(args.cache)
    manifest, budget = {}, 40_000_000_000
    api = HfApi()
    for name, repo in CATALOG.items():
        try:
            info = api.model_info(repo, files_metadata=True)
            files = [f for f in info.siblings if f.rfilename.endswith((".safetensors", ".json", ".model", ".txt", ".tiktoken", ".jinja")) and not f.rfilename.startswith("consolidated")]
            size = sum(f.size or 0 for f in files)
            if size > budget:
                raise RuntimeError(f"Download budget exceeded: {size} bytes, remaining {budget}")
            budget -= size
            manifest[name] = {"repository": repo, "revision": info.sha, "bytes": size, "status": "downloading"}
            write_json(root / "acquisition.json", manifest)
            snapshot_download(repo, revision=info.sha, local_dir=cache / name,
                              allow_patterns=[f.rfilename for f in files], max_workers=4)
            manifest[name]["status"] = "downloaded"
            print(name, "downloaded", size, flush=True)
        except Exception as exc:
            manifest.setdefault(name, {"repository": repo})
            manifest[name].update(status="failed", error=f"{type(exc).__name__}: {exc}")
            print(name, "failed", str(exc), flush=True)
        write_json(root / "acquisition.json", manifest)


if __name__ == "__main__":
    main()
