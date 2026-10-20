#!/usr/bin/env bash
# run.sh와 같은 흐름(백엔드 기동 → 헬스 대기 → UI 기동)을 컨테이너 안에서 수행한다.
# 의존성은 이미지 빌드 시 설치되므로 venv 생성 단계만 없다.
#
# 두 프로세스를 모두 백그라운드로 띄우고 `wait -n`으로 먼저 끝나는 쪽을 기다린다.
# 예전처럼 streamlit을 전경에 두면 bash가 SIGTERM을 자식에 전달하지 못해 매번
# 10초 뒤 SIGKILL로 죽었고, 백엔드가 죽어도 컨테이너는 unhealthy로만 남았다.
set -euo pipefail

BACKEND_HOST="${BACKEND_HOST:-0.0.0.0}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
UI_HOST="${UI_HOST:-0.0.0.0}"
UI_PORT="${UI_PORT:-8501}"
export WATER_API_URL="${WATER_API_URL:-http://127.0.0.1:$BACKEND_PORT}"

python -m uvicorn backend.app.main:app \
  --host "$BACKEND_HOST" --port "$BACKEND_PORT" &
backend_pid=$!

python - "http://127.0.0.1:$BACKEND_PORT/api/v1/health" <<'PY'
import json
import sys
import time
import urllib.request

url = sys.argv[1]
for _ in range(300):
    try:
        with urllib.request.urlopen(url, timeout=1) as response:
            payload = json.load(response)
        if payload.get("status") == "ok":
            break
    except Exception:
        time.sleep(0.2)
else:
    raise SystemExit(f"Backend did not become ready: {url}")
PY

echo "Backend ready: $WATER_API_URL/docs"
python -m streamlit run ui_next/app.py \
  --server.address "$UI_HOST" --server.port "$UI_PORT" &
ui_pid=$!

terminate() {
  kill -TERM "$backend_pid" "$ui_pid" 2>/dev/null || true
}
trap terminate INT TERM

# 둘 중 하나라도 끝나면(정상 종료·크래시 모두) 나머지를 정리하고 그 종료 코드로 나간다.
# 컨테이너가 죽어야 restart 정책이 다시 살린다.
set +e
wait -n "$backend_pid" "$ui_pid"
status=$?
terminate
wait "$backend_pid" "$ui_pid" 2>/dev/null
exit "$status"
