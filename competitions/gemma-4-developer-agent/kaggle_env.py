"""Build the offline evaluation wheels for Kaggle 4x L4 and upload them as a private dataset.

Kaggle offers 4x L4 only without internet. This downloads the wheels of every evaluation
package, resolved for Linux and the image's Python 3.12 with the same cutoff as validate.py,
so that validate.py can build the evaluation environment offline. The scoring sandbox runs
Python 3.13, but Kaggle never finished creating a dataset that held a CPython 3.13 build,
so tasks run on 3.12 there. Run locally once:

    uv run python competitions/gemma-4-developer-agent/kaggle_env.py
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

from validate import EVAL_PACKAGES, EXCLUDE_NEWER, WHEELHOUSE

DATASET = "hotcoffee012/kaggle-ws-eval-wheels"
PYTHON_VERSION = "3.12"  # the Kaggle image's
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
    env, stage = work / "env", work / "stage"
    for directory in (env, stage):
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True)

    harness = harness_wheels(work / "harness")
    requirements = work / "requirements.in"
    requirements.write_text("\n".join([*EVAL_PACKAGES, *map(str, harness)]) + "\n")
    pins = work / "requirements.txt"
    run(
        "uv", "pip", "compile", requirements, "-o", pins, "--no-header", "--no-annotate",
        "--python-version", PYTHON_VERSION, "--python-platform", "x86_64-manylinux_2_28",
        "--exclude-newer", EXCLUDE_NEWER, "--find-links", harness[0].parent,
    )  # fmt: skip
    # the harness wheels are listed by path; they are attached through the wheelhouse instead
    lines = [line for line in pins.read_text().splitlines() if "==" in line]
    (env / "requirements.txt").write_text("\n".join(lines) + "\n")

    command = [sys.executable, "-m", "pip", "download", "-q", "--no-deps", "--only-binary=:all:"]
    command += ["--python-version", PYTHON_VERSION, "--implementation", "cp"]
    command += ["--abi", f"cp{PYTHON_VERSION.replace('.', '')}", "--abi", "abi3", "--abi", "none"]
    command += [arg for platform in PLATFORMS for arg in ("--platform", platform)]
    run(*command, "-r", env / "requirements.txt", "-d", env)
    # Kaggle unpacks the tar; one file uploads faster than many.
    run("tar", "-cf", stage / "eval_wheels.tar", "-C", env, ".")

    metadata = {"id": DATASET, "title": DATASET.split("/")[1], "licenses": [{"name": "CC0-1.0"}]}
    (stage / "dataset-metadata.json").write_text(json.dumps(metadata))
    exists = subprocess.run(
        ["kaggle", "datasets", "status", DATASET], check=False, capture_output=True
    )
    if exists.returncode == 0:
        run("kaggle", "datasets", "version", "-p", stage, "-m", f"exclude-newer {EXCLUDE_NEWER}")
    else:
        run("kaggle", "datasets", "create", "-p", stage)
    print(f"uploaded {len(lines)} wheels for Python {PYTHON_VERSION} to {DATASET}")


if __name__ == "__main__":
    main()
