#!/usr/bin/env bash
# Upload the committed code (src/ and competitions/) as a private Kaggle dataset, so that
# kernels without internet (Kaggle 4x L4) can run it. Usage: scripts/kaggle_code.sh [message]
set -euo pipefail

DATASET="${KAGGLE_CODE_DATASET:-hotcoffee012/kaggle-ws-code}"
MESSAGE="${1:-$(git log -1 --format='%h %s')}"
ROOT="$(git rev-parse --show-toplevel)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

if ! git -C "$ROOT" diff --quiet HEAD -- src competitions; then
  echo "warning: uncommitted changes in src/ or competitions/ are not uploaded" >&2
fi
# Kaggle extracts uploaded zip files but keeps a tar as it is, so the kernel extracts it.
git -C "$ROOT" archive --format=tar -o "$STAGE/kaggle_ws.tar" HEAD src competitions
git -C "$ROOT" rev-parse HEAD > "$STAGE/COMMIT"
cat > "$STAGE/dataset-metadata.json" <<EOF
{"id": "$DATASET", "title": "kaggle-ws-code", "licenses": [{"name": "other"}]}
EOF

if uv run kaggle datasets status "$DATASET" >/dev/null 2>&1; then
  uv run kaggle datasets version -p "$STAGE" -m "$MESSAGE"
else
  uv run kaggle datasets create -p "$STAGE"
fi
echo "uploaded $(cat "$STAGE/COMMIT") to $DATASET"
