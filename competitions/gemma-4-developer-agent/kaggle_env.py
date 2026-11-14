"""Build the offline evaluation environment for Kaggle 4x L4 and upload it as a private dataset.

Kaggle offers 4x L4 only without internet, and its image has Python 3.12, while the scoring
sandbox runs Python 3.13. This downloads a standalone CPython 3.13 and the wheels of every
evaluation package (resolved for Linux with the same cutoff as validate.py), so that
validate.py can build the evaluation environment offline. Run locally once:

    uv run python competitions/gemma-4-developer-agent/kaggle_env.py
"""

import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

from validate import EVAL_PACKAGES, EXCLUDE_NEWER, WHEELHOUSE

DATASET = "hotcoffee012/gemma-4-offline-env"
PYTHON_URL = (
    "https://github.com/astral-sh/python-build-standalone/releases/download/20250212/"
    "cpython-3.13.2%2B20250212-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
)
PYTHON_ARCHIVE = "cpython-3.13-linux.tar.gz"
HARNESS = ("adk_submission", "adk_eval_core")
PLATFORMS = (
    "manylinux_2_28_x86_64",
    "manylinux_2_17_x86_64",
    "manylinux2014_x86_64",
    "manylinux_2_5_x86_64",
    "manylinux1_x86_64",
    "any",
)
REPO_ROOT = Path(__file__).resolve().parents[2]


def run(*command) -> None:
    subprocess.run([str(part) for part in command], check=True)


def harness_wheels(directory: Path) -> list[Path]:
    """The harness is not on PyPI; take it from the competition's wheelhouse dataset."""
    directory.mkdir(parents=True, exist_ok=True)
    listing = subprocess.run(
        ["kaggle", "datasets", "files", WHEELHOUSE, "--csv", "--page-size", "200"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    names = [line.split(",")[0] for line in listing.splitlines()]
    wheels = []
    for name in names:
        if name.startswith(HARNESS) and name.endswith(".whl"):
            if not (directory / name).exists():
                run("kaggle", "datasets", "download", WHEELHOUSE, "-f", name, "-p", directory, "-q")
            wheels.append(directory / name)
    return wheels


def main() -> None:
    work = REPO_ROOT / "outputs" / "gemma-4-developer-agent" / "kaggle_env"
    stage = work / "stage"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)

    harness = harness_wheels(work / "harness")
    requirements = work / "requirements.in"
    requirements.write_text("\n".join([*EVAL_PACKAGES, *map(str, harness)]) + "\n")
    pins = work / "requirements.txt"
    run(
        "uv", "pip", "compile", requirements, "-o", pins, "--no-header", "--no-annotate",
        "--python-version", "3.13", "--python-platform", "x86_64-manylinux_2_28",
        "--exclude-newer", EXCLUDE_NEWER, "--find-links", harness[0].parent,
    )  # fmt: skip
    # the harness wheels are listed by path; they are attached through the wheelhouse instead
    lines = [line for line in pins.read_text().splitlines() if "==" in line]
    (stage / "requirements.txt").write_text("\n".join(lines) + "\n")

    command = [sys.executable, "-m", "pip", "download", "-q", "--no-deps", "--only-binary=:all:"]
    command += ["--python-version", "3.13", "--implementation", "cp"]
    command += ["--abi", "cp313", "--abi", "abi3", "--abi", "none"]
    command += [arg for platform in PLATFORMS for arg in ("--platform", platform)]
    run(*command, "-r", stage / "requirements.txt", "-d", stage)

    with urllib.request.urlopen(PYTHON_URL) as response, open(stage / PYTHON_ARCHIVE, "wb") as f:
        shutil.copyfileobj(response, f)

    metadata = {"id": DATASET, "title": "gemma-4-offline-env", "licenses": [{"name": "other"}]}
    (stage / "dataset-metadata.json").write_text(json.dumps(metadata))
    exists = subprocess.run(
        ["kaggle", "datasets", "status", DATASET], check=False, capture_output=True
    )
    if exists.returncode == 0:
        run("kaggle", "datasets", "version", "-p", stage, "-m", f"exclude-newer {EXCLUDE_NEWER}")
    else:
        run("kaggle", "datasets", "create", "-p", stage)
    print(f"uploaded {len(lines)} wheels and CPython 3.13 to {DATASET}")


if __name__ == "__main__":
    main()
