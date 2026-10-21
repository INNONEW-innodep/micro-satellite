#!/usr/bin/env bash
# WATERCAST 설치 — 런타임 이미지 적재 → 앱 묶음(바이너리) 적재 → 설정 → 기동 → 상태 확인 → 스모크 테스트.
#
#   ./install.sh               기본 설치 (API·업무 UI, 데모 화면 포함)
#   ./install.sh --with-wbms   실탐지용 wbms 이미지(11 GB)까지 적재
#   ./install.sh --no-smoke    설치 후 스모크 테스트 생략
#   WITH_SMSS_NET=1 ./install.sh   3세부 공유 네트워크(smss-net)에 붙여 기동 (.env 에 적어 두면 계속 유지)
#
# 다시 실행해도 안전하다(이미 있는 .env·데이터·같은 build 의 앱은 건드리지 않는다).
# 모든 출력은 logs/install_<시각>.log 에도 남고, 실패하면 어느 단계·명령에서 멈췄는지와 볼 로그를 찍는다.
# docker 그룹이 아니면 sudo 로 실행한다 — 그 뒤의 start/stop/update_app/uninstall 도 같은 계정으로.
. "$(dirname "$0")/common.sh"

WITH_WBMS=0; RUN_SMOKE=1
for arg in "$@"; do
  case "$arg" in
    --with-wbms) WITH_WBMS=1 ;;
    --no-smoke)  RUN_SMOKE=0 ;;
    -h|--help)   sed -n '2,10p' "$0"; exit 0 ;;
    *) fail "알 수 없는 인자: $arg" ;;
  esac
done
start_log install

log "1/7 요구사항 확인"
need docker "https://docs.docker.com/engine/install/"
docker compose version >/dev/null 2>&1 || fail "docker compose v2 가 필요합니다 (docker-compose-plugin)"
need curl "curl 설치"
need python3 "python3 설치 (스모크 테스트·실탐지 러너에 사용, 표준 라이브러리만 씀)"
docker info >/dev/null 2>&1 || fail "docker 데몬에 접근할 수 없습니다 (권한: docker 그룹 또는 sudo)"
ok "docker $(docker --version | awk '{print $3}' | tr -d ,) · $(docker compose version --short)"

log "2/7 패키지 무결성 확인 (SHA256SUMS)"
if [ -f SHA256SUMS ]; then
  # 운영 중 손대는 파일(docker-compose*.yml — SELinux :z 등, .env)은 바뀌어도 통과시키고, 나머지 불일치만 실패
  bad=$(sha256sum -c SHA256SUMS 2>/dev/null | grep -v ': OK$' | grep -v '^\./docker-compose' || true)
  if [ -n "$bad" ]; then printf '%s\n' "$bad" >&2; fail "체크섬 불일치 — 전송 중 손상됐을 수 있습니다 (위 목록). 다시 받으세요"; fi
  ok "체크섬 일치"
else
  log "  SHA256SUMS 없음 — 건너뜀"
fi

log "3/7 설정 파일 (.env)"
if [ ! -f .env ]; then cp .env.example .env; ok ".env 생성 (.env.example 복사) — 포트·연계 값·키는 .env 에서 바꿉니다"; else ok ".env 있음 (그대로 사용)"; fi
load_env
mkdir -p backend/data/regions backend/data/wbms_jobs backend/data/wbms_products backend/data/wbms_fusion backend/data/results data app logs
ok "쓰기 디렉터리 backend/data/{regions,wbms_jobs,wbms_products,wbms_fusion,results} · app · logs"
printf '  설정 요약: API %s · UI %s · 데모 %s · 이미지 %s · RUN_AS %s · WBMS_GPU %s · KMA 키 %s · 연계 로그 URL %s\n' \
  "$API_PORT" "$UI_PORT" "$PIPE_PORT" "$IMAGE" "${RUN_AS:-1000:1000}" "${WBMS_GPU:-auto}" \
  "$( [ -n "${KMA_API_KEY:-}" ] && echo 있음 || echo '없음(선택)')" "$( [ -n "${BACKEND_LIAISON_LOG_URL:-}" ] && echo 있음 || echo '없음(전송 안 함)')"

log "4/7 런타임 이미지 적재"
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
  ok "$IMAGE 이미 있음"
else
  tarball=$(find images -maxdepth 1 -name 'watercast-runtime-*.tar.gz' 2>/dev/null | sort | head -1 || true)
  [ -n "$tarball" ] || fail "images/watercast-runtime-*.tar.gz 가 없습니다"
  log "  docker load -i $tarball (수 분, 진행 표시 없음)"
  docker load -i "$tarball"
  docker image inspect "$IMAGE" >/dev/null 2>&1 || fail "적재 후에도 $IMAGE 가 없습니다 — .env 의 WATERCAST_IMAGE 와 tar 안의 태그(위 'Loaded image:')를 맞추세요"
  ok "$IMAGE 적재"
fi
if [ "$WITH_WBMS" = "1" ]; then
  if docker image inspect wbms:0.13 >/dev/null 2>&1; then ok "wbms:0.13 이미 있음"
  else
    wb=$(find images -maxdepth 1 -name 'wbms-*.tar.gz' 2>/dev/null | sort | head -1 || true)
    [ -n "$wb" ] || fail "images/wbms-*.tar.gz 가 없습니다 (실탐지 묶음은 별도 전달)"
    log "  docker load -i $wb (11 GB, 10분 이상, 진행 표시 없음)"
    docker load -i "$wb"
    docker image inspect wbms:0.13 >/dev/null 2>&1 || fail "적재 후에도 wbms:0.13 이 없습니다"
    ok "wbms:0.13 적재"
  fi
fi

log "5/7 앱 묶음(바이너리) 적재 → app/current"
bundle=$(find app -maxdepth 1 -name 'watercast-app-*.tar.gz' 2>/dev/null | sort | tail -1 || true)
if [ -z "$bundle" ] && [ -d app/current ]; then ok "app/current 있음 ($(app_version)) — 묶음 파일 없이 그대로 사용"
else
  [ -n "$bundle" ] || fail "app/watercast-app-*.tar.gz 가 없습니다 (앱 묶음은 패키지에 들어 있어야 합니다)"
  install_app_bundle "$bundle" || fail "앱 묶음 적재 실패: $bundle"
fi

log "6/7 기동"
# 우리 컨테이너가 이미 포트를 잡고 있는 경우(재실행)는 통과, 남의 프로세스가 잡고 있으면 중단
ports=("$API_PORT" "$UI_PORT")
case ",${COMPOSE_PROFILES:-}," in *,demo,*) ports+=("$PIPE_PORT") ;; esac
for p in "${ports[@]}"; do
  if port_in_use "$p" && ! container_running "$NAME" && ! container_running "$NAME-pipeline"; then
    fail "호스트 포트 $p 가 이미 사용 중입니다 — .env 의 WATERCAST_*_PORT 를 바꾸세요"
  fi
done
compose_up
if wait_health "$API/api/v1/health" 90; then ok "API 응답: $API/api/v1/health"; else
  compose logs --tail 40 watercast >&2; fail "3분 안에 API 가 준비되지 않았습니다 (위 로그와 $LOG_FILE 참고)"; fi

log "7/7 확인"
if [ "$RUN_SMOKE" = "1" ]; then ./smoke_test.sh; fi
cat <<MSG

설치 완료  (앱 $(app_version))
  API 문서   http://<이 서버>:${API_PORT}/docs        (인증 없음 — 바로 호출)
  업무 UI    http://<이 서버>:${UI_PORT}/
  간단 화면  http://<이 서버>:${API_PORT}/simple
  데모 UI    http://<이 서버>:${PIPE_PORT}/   (COMPOSE_PROFILES=demo 일 때)
  배포 후 확인  ./postcheck.sh     상태 ./status.sh     중지 ./stop.sh     재기동 ./start.sh
  앱 교체    ./update_app.sh app/watercast-app-<새 build>.tar.gz   (이미지는 그대로)
  연계 값    .env 의 BACKEND_LIAISON_* 를 채우고 ./start.sh
  로그       $LOG_FILE
MSG
