#!/usr/bin/env bash
# One command for every offline check in this repository (SB-4).
# Prints one line per step: PASS: <step>, FAIL: <step> or UNCHECKED: <step>.
# A step whose tool or file is missing prints UNCHECKED and does not fail the run.
# Exit 0 when no step failed, 1 otherwise. Never touches the network or AWS.
# Usage: bash devtools/check.sh   (from anywhere; it cd's to the repo root)
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PYTHON:-python}"
command -v "$PY" >/dev/null 2>&1 || PY=python3
LOG="$(mktemp)"
trap 'rm -f "$LOG"' EXIT
failed=0

# run_step NAME NEEDS CMD...
#   NEEDS is a path, a command name, or py:<module> (importable by $PY).
#   Missing NEEDS prints UNCHECKED: NAME (<needs> not found) and skips CMD.
run_step() {
  local name="$1" needs="$2"
  shift 2
  local have=1
  case "$needs" in
    py:*) "$PY" -c "import ${needs#py:}" >/dev/null 2>&1 || have=0 ;;
    */*|*.*) [ -e "$needs" ] || have=0 ;;
    *) command -v "$needs" >/dev/null 2>&1 || have=0 ;;
  esac
  if [ "$have" -eq 0 ]; then
    echo "UNCHECKED: $name ($needs not found)"
    return 0
  fi
  if "$@" >"$LOG" 2>&1; then
    echo "PASS: $name"
  else
    tail -n 40 "$LOG" >&2
    echo "FAIL: $name"
    failed=1
  fi
}

# Before SB-2 adds [tool.pytest.ini_options], the two legacy tests that cannot
# import (test_sunback.py, test_parameters.py) are skipped by path.
if grep -q '^\[tool\.pytest\.ini_options\]' pyproject.toml; then
  run_step pytest py:pytest "$PY" -m pytest -q
else
  run_step pytest py:pytest "$PY" -m pytest -q sunback/__tests__ \
    --ignore=sunback/__tests__/test_sunback.py \
    --ignore=sunback/__tests__/test_parameters.py
fi

run_step compileall "$PY" "$PY" -m compileall -q -x '(/dep/|/depricated/)' sunback aws_lambda

# ruff runs only once SB-2 adds a [tool.ruff] table; without it ruff's defaults
# lint the whole legacy tree, which is not this check's job.
if grep -q '^\[tool\.ruff\]' pyproject.toml; then
  run_step ruff ruff ruff check
else
  echo "UNCHECKED: ruff (no [tool.ruff] in pyproject.toml yet; SB-2 adds it)"
fi

run_step catalog sunback/__tests__/test_product_catalog.py \
  "$PY" -m pytest -q sunback/__tests__/test_product_catalog.py

exit "$failed"
