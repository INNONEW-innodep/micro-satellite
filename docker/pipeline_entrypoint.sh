#!/usr/bin/env bash
# 발표용 파이프라인 데모 UI(pipeline_ui). 백엔드 API를 쓰지 않는 독립 Streamlit 앱이다.
set -euo pipefail

UI_HOST="${UI_HOST:-0.0.0.0}"
UI_PORT="${UI_PORT:-8501}"

exec python -m streamlit run pipeline_ui/app.py \
  --server.address "$UI_HOST" --server.port "$UI_PORT"
