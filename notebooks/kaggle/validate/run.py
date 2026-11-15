"""Run validate.py from the kaggle_ws repo on Kaggle 4x L4, the scoring hardware.

Upload the code with `scripts/kaggle_code.sh`, then push with
`uv run kaggle kernels push -p notebooks/kaggle/validate`. Results go to
/kaggle/working/validate/; to continue them, attach this kernel's previous output as an input.
Adapters come from attached inputs that hold <exp>/<adapter>/, such as the train kernel's output.
"""

import os
import shlex
import subprocess
import sys
from pathlib import Path

COMPETITION = "gemma-4-developer-agent"
ARGS = "--exps exp001,exp003,exp005"  # the first is the baseline; drop --num-tasks for 30
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
script = f"{REPO_DIR}/competitions/{COMPETITION}/validate.py"
subprocess.run([sys.executable, script, *shlex.split(ARGS)], check=True, env=env)
