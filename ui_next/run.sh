#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${WATER_APP_VENV:-$PROJECT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
STAMP_FILE="$VENV_DIR/.ui-requirements-installed"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

if [[ ! -f "$STAMP_FILE" || "$SCRIPT_DIR/requirements.txt" -nt "$STAMP_FILE" ]]; then
  "$VENV_DIR/bin/python" -m pip install --upgrade pip
  "$VENV_DIR/bin/python" -m pip install -r "$SCRIPT_DIR/requirements.txt"
  touch "$STAMP_FILE"
fi

cd "$PROJECT_DIR"
"$VENV_DIR/bin/python" -m streamlit run "$SCRIPT_DIR/app.py" \
  --server.address "${UI_HOST:-0.0.0.0}" --server.port "${UI_PORT:-8501}"
