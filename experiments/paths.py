"""Keep experiment inputs and outputs outside the source checkout."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path(os.environ.get(
    "LOCAL_ANNOTATOR_ARTIFACTS", PROJECT_ROOT.parent / "local_annotator_artifacts"
)).expanduser().resolve()
if ARTIFACT_ROOT == PROJECT_ROOT or PROJECT_ROOT in ARTIFACT_ROOT.parents:
    raise ValueError("LOCAL_ANNOTATOR_ARTIFACTS must be outside the source checkout")
DATA = ARTIFACT_ROOT / "data"
MEASUREMENTS = ARTIFACT_ROOT / "measurements"
