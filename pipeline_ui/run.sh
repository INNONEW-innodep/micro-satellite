#!/usr/bin/env bash
# 발표용 데모 UI 실행 스크립트
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# 가상환경 (선택) — 없으면 사용자 환경에 설치
if [ ! -d ".venv" ]; then
  echo "[setup] .venv 생성 중..."
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "[setup] 의존성 설치 중..."
pip install --upgrade pip >/dev/null
pip install -r requirements.txt

echo ""
echo "============================================================"
echo "  🛰️  파이프라인 데모 UI 시작"
echo "  브라우저에서 자동으로 열립니다. (열리지 않으면 아래 URL 접속)"
echo "============================================================"
echo ""

streamlit run app.py --server.port 8501 --server.address 0.0.0.0
