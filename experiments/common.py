import json
from pathlib import Path


def artifact_root(value):
    root = Path(value).expanduser().resolve()
    source = Path(__file__).resolve().parents[1]
    if root == source or source in root.parents:
        raise ValueError("Artifacts must be outside the source checkout")
    root.mkdir(parents=True, exist_ok=True)
    return root


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
