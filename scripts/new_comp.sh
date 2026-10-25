#!/usr/bin/env bash
# Usage: scripts/new_comp.sh <competition-slug>
set -euo pipefail
comp="${1:?usage: new_comp.sh <competition-slug>}"
root="$(cd "$(dirname "$0")/.." && pwd)"
dst="$root/competitions/$comp"
[ -e "$dst" ] && { echo "already exists: $dst"; exit 1; }
cp -r "$root/competitions/_template" "$dst"
sed "s/COMPETITION_NAME/$comp/g" "$dst/README.md" > "$dst/README.md.tmp" && mv "$dst/README.md.tmp" "$dst/README.md"
echo "created: $dst"
