#!/usr/bin/env bash
# Sequential sweeps of the paper designs (one at a time to stay inside the RAM budget).
set -u
cd "$(dirname "$0")/.."
for d in "$@"; do
  echo "=== $d $(date -Is)"
  .venv/bin/python -m fem.sweep --design "$d" --quiet 2>&1 | grep -v Warning
done
