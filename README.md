# kaggle_ws

Kaggle workspace: write code locally, push to GitHub, then clone and run it on Colab.

## Layout

```
kaggle_ws/
├── src/kaggle_ws/          # shared utilities (importable after `pip install -e .`)
│   ├── paths.py            #   resolves data / output paths for local, Colab and Kaggle
│   └── utils.py            #   seed_everything, timer, ...
├── competitions/
│   ├── _template/          # skeleton for a new competition
│   └── <competition>/
│       ├── README.md       #   overview and experiment log (CV / LB)
│       ├── notebooks/      #   EDA
│       └── exp/
│           ├── exp001/train.py
│           └── exp002/train.py
├── notebooks/
│   └── colab_runner.ipynb  # Colab runner template
├── scripts/
│   ├── new_comp.sh         # create a competition directory
│   ├── new_exp.sh          # copy the latest experiment and bump the number
│   └── download.sh         # download and unzip data with the Kaggle CLI
├── data/                   # (git-ignored) local data
└── outputs/                # (git-ignored) local outputs
```

- One experiment = one directory. Don't edit past experiments; copy them to start the next one (reproducibility).
- Competition and experiment names are derived from the directory names, and outputs go to `outputs/<competition>/<exp>/`.
- On Colab, outputs go to `MyDrive/kaggle/outputs/<competition>/<exp>/` when Google Drive is mounted.

## Local setup

```bash
uv sync                      # create .venv and install dependencies (including the Kaggle CLI)
source .venv/bin/activate    # or prefix each command with `uv run`
```

Kaggle API authentication (either one):

- New: Kaggle > Settings > API > **Generate New Token**, then `export KAGGLE_API_TOKEN=...` (or save it to `~/.kaggle/access_token`)
- Legacy: put the `kaggle.json` from **Create Legacy API Key** at `~/.kaggle/kaggle.json` and `chmod 600` it

```bash
kaggle competitions list     # check that it works
```

## Usage

```bash
# 1. Create a competition directory (accept the competition rules on Kaggle first)
scripts/new_comp.sh titanic

# 2. Optionally download the data and do a quick local check
scripts/download.sh titanic
uv run python competitions/titanic/exp/exp001/train.py --debug

# 3. Push
git add -A && git commit -m "titanic exp001" && git push

# 4. Open notebooks/colab_runner.ipynb on Colab, edit the settings cell, and run all cells
#    https://colab.research.google.com/github/Moriyuki-S/kaggle_ws/blob/main/notebooks/colab_runner.ipynb

# Next experiment
scripts/new_exp.sh titanic   # copies exp001 -> exp002
```

## Task shortcuts (local only)

[Task](https://taskfile.dev) (`brew install go-task`) wraps the commands above. Colab does not use it; the notebook calls `scripts/` and `train.py` directly.

```bash
task                                   # list tasks
task new-comp COMP=titanic
task new-exp  COMP=titanic             # exp001 -> exp002
task download COMP=titanic
task debug    COMP=titanic EXP=exp002
task run      COMP=titanic EXP=exp002 -- --seed 0   # extra args after --
task submit   COMP=titanic EXP=exp002 MSG="lgbm baseline"
task lint / task format
task colab                             # print the Colab URL
```

## Colab first-time setup

Register these in the 🔑 Secrets sidebar and enable notebook access:

- `KAGGLE_API_TOKEN` (or `KAGGLE_USERNAME` and `KAGGLE_KEY` for the legacy key)
- `GITHUB_TOKEN`: only for private repos (fine-grained PAT, Contents: Read-only)
