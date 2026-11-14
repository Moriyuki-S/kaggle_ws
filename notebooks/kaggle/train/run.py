"""Run an experiment's train.py from the kaggle_ws repo on a Kaggle GPU machine.

Push with `uv run kaggle kernels push -p notebooks/kaggle/train`. Outputs go to
/kaggle/working/<EXP>/; to resume, attach this kernel's previous output as an input.
"""

import os
import shlex
import subprocess
import sys

COMPETITION = "gemma-4-developer-agent"
EXP = "exp003"
ARGS = "--smoke"  # "--smoke" measures speed and memory; "" trains
BRANCH = "main"
REPO = "https://github.com/Moriyuki-S/kaggle_ws.git"
REPO_DIR = "/tmp/kaggle_ws"

try:
    from kaggle_secrets import UserSecretsClient

    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:  # noqa: BLE001  (a missing secret raises a generic error)
    print("no HF_TOKEN in Kaggle secrets; downloads from Hugging Face are anonymous")

if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", "-q", "-b", BRANCH, REPO, REPO_DIR], check=True)
subprocess.run(["git", "-C", REPO_DIR, "log", "-1", "--oneline"], check=True)

# kaggle_ws is used from source: installing the repo would replace Kaggle's torch.
env = {**os.environ, "PYTHONPATH": f"{REPO_DIR}/src"}
script = f"{REPO_DIR}/competitions/{COMPETITION}/exp/{EXP}/train.py"
subprocess.run([sys.executable, script, *shlex.split(ARGS)], check=True, env=env)
