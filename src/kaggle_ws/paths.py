"""Resolve data / output directories for local, Colab and Kaggle Notebook."""

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MY_DRIVE = Path("/content/drive/MyDrive")
DEFAULT_DRIVE_ROOT = "Colab Notebooks/Kaggle"


def is_colab() -> bool:
    return "COLAB_RELEASE_TAG" in os.environ or Path("/content").exists()


def is_kaggle() -> bool:
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ


@dataclass(frozen=True)
class Paths:
    data: Path
    output: Path


def drive_root() -> Path:
    """The folder in My Drive that holds one folder per competition, named after it.

    DRIVE_ROOT overrides the default and may also be written as a full or "My Drive/" path.
    """
    folder = os.environ.get("DRIVE_ROOT", DEFAULT_DRIVE_ROOT).strip().strip("/")
    for prefix in ("content/drive/MyDrive/", "My Drive/", "マイドライブ/"):
        folder = folder.removeprefix(prefix)
    return MY_DRIVE / folder


def get_paths(competition: str) -> Paths:
    """DATA_DIR / OUTPUT_DIR env vars take precedence."""
    if is_kaggle():
        data = Path("/kaggle/input") / competition
        output = Path("/kaggle/working")
    elif is_colab():
        data = Path("/content/data") / competition
        # keep outputs on Google Drive if it is mounted
        output = (drive_root() if MY_DRIVE.exists() else Path("/content/outputs")) / competition
    else:
        data = REPO_ROOT / "data" / competition
        output = REPO_ROOT / "outputs" / competition

    data = Path(os.environ.get("DATA_DIR", data))
    output = Path(os.environ.get("OUTPUT_DIR", output))
    output.mkdir(parents=True, exist_ok=True)
    return Paths(data=data, output=output)
