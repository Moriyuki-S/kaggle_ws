"""Run an experiment's train.py from the kaggle_ws repo on a Kaggle GPU machine.

Upload the code with `scripts/kaggle_code.sh`, then push with
`uv run kaggle kernels push -p notebooks/kaggle/train`. Outputs go to
/kaggle/working/<EXP>/; to resume, attach this kernel's previous output as an input.
"""

import os
import shlex
import subprocess
import sys
from pathlib import Path

COMPETITION = "gemma-4-developer-agent"
EXP = "exp003"
ARGS = "--time-limit-hours 10.5"  # "--smoke" measures speed and memory; Kaggle stops a session at 12 h
REPO_DIR = "/tmp/kaggle_ws"

# 4x L4 runs without internet, so the code comes from the kaggle-ws-code dataset.
dataset = next(Path("/kaggle/input").glob("**/kaggle-ws-code"), None)
if dataset is None:
    raise SystemExit("Attach the kaggle-ws-code dataset (scripts/kaggle_code.sh).")
archive = next(dataset.rglob("kaggle_ws.tar"), None)
if archive is not None:
    os.makedirs(REPO_DIR, exist_ok=True)
    subprocess.run(["tar", "-xf", str(archive), "-C", REPO_DIR], check=True)
else:  # Kaggle unpacked the archive
    REPO_DIR = str(next(dataset.rglob("src/kaggle_ws")).parents[1])
print("code at commit", next(dataset.rglob("COMMIT")).read_text().strip(), flush=True)

# kaggle_ws is used from source: installing the repo would replace Kaggle's torch.
env = {**os.environ, "PYTHONPATH": f"{REPO_DIR}/src"}
script = f"{REPO_DIR}/competitions/{COMPETITION}/exp/{EXP}/train.py"
subprocess.run([sys.executable, script, *shlex.split(ARGS)], check=True, env=env)
