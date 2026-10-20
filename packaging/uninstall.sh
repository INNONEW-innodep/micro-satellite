#!/usr/bin/env bash
# 제거 — 컨테이너를 내리고, 옵션에 따라 이미지·데이터도 지운다.
#   ./uninstall.sh                컨테이너만
#   ./uninstall.sh --images       + watercast 이미지
#   ./uninstall.sh --wbms-image   + wbms:0.13 이미지 (11 GB, 다른 작업과 공유될 수 있어 따로 지정)
#   ./uninstall.sh --data         + backend/data (잡 상태·규약 산출물·사용자 지역 — 되돌릴 수 없음)
# 컨테이너가 만든 파일은 root 소유라 --data 는 설치 때와 같은 계정(sudo 면 sudo)으로 실행한다.
. "$(dirname "$0")/common.sh"; load_env
RM_IMAGES=0; RM_WBMS=0; RM_DATA=0
for arg in "$@"; do case "$arg" in --images) RM_IMAGES=1;; --wbms-image) RM_WBMS=1;; --data) RM_DATA=1;; -h|--help) sed -n '2,7p' "$0"; exit 0;; *) fail "알 수 없는 인자: $arg";; esac; done
compose_down || true
ok "컨테이너 제거"
if [ "$RM_IMAGES" = "1" ]; then docker image rm -f "$IMAGE" 2>/dev/null || true; ok "이미지 제거: $IMAGE"; fi
if [ "$RM_WBMS" = "1" ]; then docker image rm -f wbms:0.13 2>/dev/null || true; ok "이미지 제거: wbms:0.13"; fi
if [ "$RM_DATA" = "1" ]; then
  if rm -rf backend/data 2>/dev/null; then ok "backend/data 제거"
  else fail "backend/data 를 지우지 못했습니다 — 컨테이너가 만든 root 소유 파일입니다. sudo ./uninstall.sh --data"; fi
fi
