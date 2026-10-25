#!/usr/bin/env bash
# Usage: scripts/download.sh <competition-slug> [dest_dir]
# Accept the competition rules on Kaggle first, otherwise you get 403
set -euo pipefail
comp="${1:?usage: download.sh <competition-slug> [dest_dir]}"
root="$(cd "$(dirname "$0")/.." && pwd)"
dest="${2:-$root/data/$comp}"
mkdir -p "$dest"
kaggle competitions download -c "$comp" -p "$dest"
unzip -q -o "$dest/$comp.zip" -d "$dest" && rm "$dest/$comp.zip"
ls "$dest"
