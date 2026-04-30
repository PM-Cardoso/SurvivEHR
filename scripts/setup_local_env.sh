#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
FAST_EHR_ROOT="${FASTEHR_ROOT:-$WORKSPACE_ROOT/FastEHR-main}"
VENV_PATH="${VENV_PATH:-$REPO_ROOT/.venv}"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3.11)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "Could not find python3.11 or python3 on PATH." >&2
    exit 1
  fi
fi

PY_VERSION="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$PY_VERSION" != "3.10" && "$PY_VERSION" != "3.11" ]]; then
  echo "SurvivEHR currently needs Python 3.10 or 3.11; found $PY_VERSION at $PYTHON_BIN." >&2
  echo "Set PYTHON_BIN to a compatible interpreter, e.g. PYTHON_BIN=$(command -v python3.11)." >&2
  exit 1
fi

if [[ ! -d "$FAST_EHR_ROOT/FastEHR" ]]; then
  echo "FastEHR package directory not found at $FAST_EHR_ROOT." >&2
  echo "Set FASTEHR_ROOT to your local FastEHR checkout, e.g. export FASTEHR_ROOT=/path/to/FastEHR-main" >&2
  exit 1
fi

"$PYTHON_BIN" -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"
python -m pip install --upgrade pip "setuptools<81" wheel
python -m pip install -r "$REPO_ROOT/requirements.txt"
pushd "$FAST_EHR_ROOT" >/dev/null
python -m pip install -r requirements-py311.txt
python -m pip install -e .
popd >/dev/null

cat <<EOF

Environment ready.

Activate it with:
source "$VENV_PATH/bin/activate"

Optional environment variables:
export FASTEHR_ROOT="$FAST_EHR_ROOT"
export SURVIVEHR_DB_PATH="/absolute/path/to/cprd.db"
export SURVIVEHR_DS_PATH="/absolute/path/to/pretrain_dataset"
export SURVIVEHR_META_PATH="/absolute/path/to/meta_information.pickle"
EOF
