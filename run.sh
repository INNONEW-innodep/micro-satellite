#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${WATER_APP_VENV:-$PROJECT_DIR/.venv}"
STAMP_FILE="$VENV_DIR/.requirements-installed"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

needs_install=false
if [[ ! -f "$STAMP_FILE" ]]; then
  needs_install=true
elif find "$PROJECT_DIR" -maxdepth 2 -name 'requirements*.txt' -newer "$STAMP_FILE" -print -quit | grep -q .; then
  needs_install=true
fi

if [[ "$needs_install" == true ]]; then
  "$VENV_DIR/bin/python" -m pip install --upgrade pip
  "$VENV_DIR/bin/python" -m pip install -r "$PROJECT_DIR/requirements.txt"
  touch "$STAMP_FILE"
fi

BACKEND_HOST="${BACKEND_HOST:-0.0.0.0}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
UI_HOST="${UI_HOST:-0.0.0.0}"
UI_PORT="${UI_PORT:-8501}"
BACKEND_CLIENT_HOST="${BACKEND_CLIENT_HOST:-127.0.0.1}"
export WATER_API_URL="${WATER_API_URL:-http://$BACKEND_CLIENT_HOST:$BACKEND_PORT}"

cd "$PROJECT_DIR"
"$VENV_DIR/bin/python" -m uvicorn backend.app.main:app \
  --host "$BACKEND_HOST" --port "$BACKEND_PORT" &
backend_pid=$!

cleanup() {
  kill "$backend_pid" 2>/dev/null || true
  wait "$backend_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

"$VENV_DIR/bin/python" - "$WATER_API_URL/api/v1/health" <<'PY'
import json
import sys
import time
import urllib.request

url = sys.argv[1]
for _ in range(80):
    try:
        with urllib.request.urlopen(url, timeout=1) as response:
            payload = json.load(response)
        if payload.get("status") == "ok":
            break
    except Exception:
        time.sleep(0.1)
else:
    raise SystemExit(f"Backend did not become ready: {url}")
PY

echo "Backend: $WATER_API_URL/docs"
echo "UI:      http://localhost:$UI_PORT"
"$VENV_DIR/bin/python" -m streamlit run "$PROJECT_DIR/ui_next/app.py" \
  --server.address "$UI_HOST" --server.port "$UI_PORT"
