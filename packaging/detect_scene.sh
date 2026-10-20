#!/usr/bin/env bash
# 실탐지 — 호스트에서 wbms:0.13 이미지로 장면 하나를 탐지하고 규약 파일명 산출물·로그를 만든다.
# (API 컨테이너가 docker 소켓으로 직접 탐지하지만, 소켓을 못 쓰는 환경을 위해 호스트 러너도 남겨 둔다.)
#
#   ./detect_scene.sh 20200302                        ICEYE 부산 2020-03-02 (CPU, 10분 내외)
#   ./detect_scene.sh 20200302 --gpu auto             GPU 있으면 GPU (30초 내외), 부족하면 CPU 폴백
#   ./detect_scene.sh 20200302 --order-id 1 --scene-id 1 --work-id 1 --product-id ICEYE_X5_SLC_SM_23467_20200302T183857
#   ./detect_scene.sh 20200302 --force               이미 탐지된(캐시) 장면을 다시 탐지. 없으면 완료 마커를 보고 건너뛴다
#
# 필요: images/wbms-0.13.tar.gz 적재(install.sh --with-wbms), data/incoming/handover 의 03_model·05_l1_pre·06_aux
#       (실탐지 묶음에 포함). 산출: data/wbms_runs/wb/<yyyy>/<date>/WB_*.tif 와 backend/data/wbms_products/PT000003_*.tif
# 설치를 sudo 로 했으면 이것도 sudo 로 (backend/data/wbms_products 가 root 소유).
. "$(dirname "$0")/common.sh"; load_env
case "${1:-}" in ""|-h|--help) sed -n '2,12p' "$0"; exit 1;; esac
DATE="$1"; shift
[[ "$DATE" =~ ^[0-9]{8}$ ]] || fail "날짜는 YYYYMMDD 8자리입니다: $DATE"
docker image inspect wbms:0.13 >/dev/null 2>&1 || fail "wbms:0.13 이미지가 없습니다 — ./install.sh --with-wbms"
[ -d data/incoming/handover/03_model ] || fail "data/incoming/handover/03_model 이 없습니다 (실탐지 묶음 필요)"
[ -f src/scripts/wbms_chain_runner.py ] || fail "src/scripts/wbms_chain_runner.py 가 없습니다 (패키지가 불완전)"
export BACKEND_LIAISON_LOG_URL="${BACKEND_LIAISON_LOG_URL:-}" \
       BACKEND_LIAISON_SYSTEM_CODE="${BACKEND_LIAISON_SYSTEM_CODE:-SC000000}" \
       BACKEND_LIAISON_ALGM_ID="${BACKEND_LIAISON_ALGM_ID:-AG000000}"
export WBMS_HANDOVER_DIR="$HERE/data/incoming/handover"
mkdir -p "$HERE/data/wbms_runs" "$HERE/backend/data/wbms_products"
cd "$HERE/src" && exec python3 scripts/wbms_chain_runner.py --scenes "$DATE" --steps wb \
  --out-root "$HERE/data/wbms_runs" --product-dir "$HERE/backend/data/wbms_products" "$@"
