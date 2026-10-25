"""Experiment script. Competition / experiment names come from the directory names.

    python competitions/<competition>/exp/exp001/train.py
    -> outputs/<competition>/exp001/ (on Google Drive when run on Colab)
"""

import argparse
from pathlib import Path

import pandas as pd

from kaggle_ws import get_paths, seed_everything, timer

EXP_DIR = Path(__file__).resolve().parent
EXP = EXP_DIR.name
COMPETITION = EXP_DIR.parents[1].name


class CFG:
    seed = 42
    n_folds = 5
    target = "target"
    debug = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--debug", action="store_true", help="quick run on a small subset")
    return p.parse_args()


def main():
    args = parse_args()
    CFG.debug = args.debug
    seed_everything(CFG.seed)
    paths = get_paths(COMPETITION)
    out_dir = paths.output / EXP
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"competition: {COMPETITION} / exp: {EXP}\ndata: {paths.data}\noutput: {out_dir}")

    with timer("load"):
        print(sorted(p.name for p in paths.data.iterdir()))
        # train = pd.read_csv(paths.data / "train.csv")
        # test = pd.read_csv(paths.data / "test.csv")
        sample = pd.read_csv(paths.data / "sample_submission.csv")

    # TODO: features -> CV training (save OOF to out_dir / "oof.csv") -> inference

    sample.to_csv(out_dir / "submission.csv", index=False)
    print(f"saved: {out_dir / 'submission.csv'}")


if __name__ == "__main__":
    main()
