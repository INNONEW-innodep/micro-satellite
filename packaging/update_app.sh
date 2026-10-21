#!/usr/bin/env bash
# 앱 묶음(바이너리) 교체 — 런타임 이미지는 그대로 두고 코드만 바꾼다. 수십 초.
#
#   ./update_app.sh app/watercast-app-<버전>-<빌드>.tar.gz    새 묶음 적재 → 재기동 → 헬스 → 스모크
#   ./update_app.sh --rollback                                 직전 build(app/prev) 로 되돌리고 재기동
#   ./update_app.sh ... --no-smoke                             스모크 테스트 생략
#
# 묶음 옆에 <이름>.sha256 이 있으면 먼저 검증한다. 의존성(requirements)이 바뀐 묶음은 런타임 이미지도 새로 받아
# .env 의 WATERCAST_IMAGE 를 바꾼 뒤 ./install.sh 를 다시 돌린다(묶음의 VERSION.json runtime_image 가 그 태그다).
. "$(dirname "$0")/common.sh"; load_env
TARBALL=""; ROLLBACK=0; RUN_SMOKE=1
for arg in "$@"; do
  case "$arg" in
    --rollback) ROLLBACK=1 ;;
    --no-smoke) RUN_SMOKE=0 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) TARBALL="$arg" ;;
  esac
done
start_log update_app
if [ "$ROLLBACK" = "1" ]; then
  [ -d app/prev ] || fail "되돌릴 이전 build(app/prev)가 없습니다"
  log "1/3 되돌리기: app/current ↔ app/prev"
  mv app/current app/.swap && mv app/prev app/current && mv app/.swap app/prev
  ok "되돌림: $(app_version)"
else
  [ -n "$TARBALL" ] || fail "새 앱 묶음 경로를 주세요: ./update_app.sh app/watercast-app-<버전>-<빌드>.tar.gz"
  case "$TARBALL" in /*) ;; *) TARBALL="$HERE/$TARBALL" ;; esac
  log "1/3 앱 묶음 적재: $(basename "$TARBALL")   (현재: $(app_version))"
  wanted=$(python3 - "$TARBALL" <<'PY' 2>/dev/null || true
import json, sys, tarfile
with tarfile.open(sys.argv[1]) as t:
    for m in t.getmembers():
        if m.name.endswith("/VERSION.json") and m.name.count("/") == 1:
            print(json.load(t.extractfile(m)).get("runtime_image", "")); break
PY
)
  if [ -n "$wanted" ] && [ "$wanted" != "$IMAGE" ]; then
    warn "이 묶음은 런타임 이미지 $wanted 용인데 지금 설정은 $IMAGE 입니다 — 의존성이 바뀐 묶음이면 새 런타임 이미지를 받아 .env 의 WATERCAST_IMAGE 를 바꾸고 ./install.sh"
  fi
  install_app_bundle "$TARBALL" || fail "앱 묶음 적재 실패 — 기존 app/current 는 그대로입니다"
fi

log "2/3 재기동 (이미지 그대로, 코드만 다시 읽음)"
compose up -d --force-recreate --remove-orphans
if wait_health "$API/api/v1/health" 90; then ok "API 응답: $API/api/v1/health"; else
  compose logs --tail 40 watercast >&2
  fail "3분 안에 API 가 준비되지 않았습니다 — 되돌리기: ./update_app.sh --rollback (위 로그와 $LOG_FILE 참고)"
fi
served=$(curl -fsS --max-time 5 "$API/api/v1/health" | python3 -c 'import json,sys; b=(json.load(sys.stdin).get("build") or {}); print(b.get("build","-"))' 2>/dev/null || echo "-")
ok "API 가 보고하는 앱 build: $served"

log "3/3 확인"
if [ "$RUN_SMOKE" = "1" ]; then ./smoke_test.sh || fail "스모크 테스트 실패 — 되돌리기: ./update_app.sh --rollback"; fi
ok "앱 교체 완료: $(app_version)   로그 $LOG_FILE"
