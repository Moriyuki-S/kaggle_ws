#!/usr/bin/env bash
# Usage: scripts/new_exp.sh <competition-slug> [source exp (default: latest)]
set -euo pipefail
comp="${1:?usage: new_exp.sh <competition-slug> [source exp]}"
root="$(cd "$(dirname "$0")/.." && pwd)"
exp_root="$root/competitions/$comp/exp"
src="${2:-$(ls -d "$exp_root"/exp[0-9]* | sort | tail -n1 | xargs basename)}"
last="$(ls -d "$exp_root"/exp[0-9]* | sort | tail -n1 | xargs basename)"
next="$(printf 'exp%03d' $((10#${last#exp} + 1)))"
cp -r "$exp_root/$src" "$exp_root/$next"
echo "created: $exp_root/$next (from $src)"
