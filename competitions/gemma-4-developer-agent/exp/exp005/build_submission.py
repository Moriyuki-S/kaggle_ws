"""Validate this experiment's agent config and package it as submission.zip.

Run with the harness environment from the competition directory:
    .venv/bin/python exp/exp005/build_submission.py
"""

import zipfile
from pathlib import Path

from adk_submission import validate_directory
from adk_submission.discovery import discover_declared_models
from swegemma.config import ALLOWED_SUBMISSION_EXTENSIONS, build_submission_limits

ALLOWED_MODELS = {
    "gemma-4-31b-it-qat-w4a16-ct",
    "gemma-4-12b-it-qat-w4a16-ct",
    "gemma-4-e4b-it-qat-w4a16-ct",
    "gemma-4-e2b-it-qat-w4a16-ct",
}

EXP_DIR = Path(__file__).resolve().parent
SUBMISSION_DIR = EXP_DIR / "submission"
ZIP_PATH = EXP_DIR.parents[1] / "outputs" / EXP_DIR.name / "submission.zip"


def main() -> None:
    limits, _ = build_submission_limits()
    validate_directory(SUBMISSION_DIR, limits=limits)

    models = set(discover_declared_models(SUBMISSION_DIR))
    if len(models) != 1 or not models <= ALLOWED_MODELS:
        raise ValueError(f"expected one allowed model, got {sorted(models)}")

    files = sorted(p for p in SUBMISSION_DIR.rglob("*") if p.is_file() and p.name != ".DS_Store")
    for path in files:
        if path.suffix.lower() not in ALLOWED_SUBMISSION_EXTENSIONS:
            raise ValueError(f"disallowed extension: {path}")

    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(SUBMISSION_DIR))
    print(f"{ZIP_PATH} ({len(files)} files, model={models.pop()})")


if __name__ == "__main__":
    main()
