#!/usr/bin/env bash
# WATERCAST 설치 — 이미지 적재 → 설정 → 기동 → 상태 확인 → 스모크 테스트.
#
#   ./install.sh               기본 설치 (API·UI, 데모 화면 포함)
#   ./install.sh --with-wbms   실탐지용 wbms 이미지(11 GB)까지 적재
#   ./install.sh --no-smoke    설치 후 스모크 테스트 생략
#   WITH_SMSS_NET=1 ./install.sh   3세부 공유 네트워크(smss-net)에 붙여 기동 (.env 에 적어 두면 계속 유지)
#
# 다시 실행해도 안전하다(이미 있는 .env·데이터는 건드리지 않는다).
# docker 그룹이 아니면 sudo 로 실행한다 — 그 뒤의 start/stop/uninstall 도 같은 계정으로.
. "$(dirname "$0")/common.sh"

WITH_WBMS=0; RUN_SMOKE=1
for arg in "$@"; do
  case "$arg" in
    --with-wbms) WITH_WBMS=1 ;;
    --no-smoke)  RUN_SMOKE=0 ;;
    -h|--help)   sed -n '2,9p' "$0"; exit 0 ;;
    *) fail "알 수 없는 인자: $arg" ;;
  esac
done

log "1/6 요구사항 확인"
need docker "https://docs.docker.com/engine/install/"
docker compose version >/dev/null 2>&1 || fail "docker compose v2 가 필요합니다 (docker-compose-plugin)"
need curl "curl 설치"
need python3 "python3 설치 (스모크 테스트·실탐지 러너에 사용, 표준 라이브러리만 씀)"
docker info >/dev/null 2>&1 || fail "docker 데몬에 접근할 수 없습니다 (권한: docker 그룹 또는 sudo)"
ok "docker $(docker --version | awk '{print $3}' | tr -d ,) · $(docker compose version --short)"

log "2/6 패키지 무결성 확인 (SHA256SUMS)"
if [ -f SHA256SUMS ]; then
  # 운영 중 손대는 파일(docker-compose*.yml — SELinux :z 등)은 바뀌어도 통과시키고, 나머지 불일치만 실패
  bad=$(sha256sum -c SHA256SUMS 2>/dev/null | grep -v ': OK$' | grep -v '^\./docker-compose' || true)
  if [ -n "$bad" ]; then printf '%s\n' "$bad" >&2; fail "체크섬 불일치 — 전송 중 손상됐을 수 있습니다 (위 목록)"; fi
  ok "체크섬 일치"
else
  log "  SHA256SUMS 없음 — 건너뜀"
fi

log "3/6 설정 파일"
if [ ! -f .env ]; then cp .env.example .env; ok ".env 생성 (.env.example 복사) — 포트·연계 값은 .env 에서 바꿉니다"; else ok ".env 유지"; fi
load_env
mkdir -p backend/data/regions backend/data/wbms_jobs backend/data/wbms_products backend/data/wbms_fusion backend/data/results data
ok "쓰기 디렉터리 backend/data/{regions,wbms_jobs,wbms_products,wbms_fusion,results}"

log "4/6 이미지 적재"
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
  ok "$IMAGE 이미 있음"
else
  tarball=$(find images -maxdepth 1 -name 'watercast-*.tar.gz' 2>/dev/null | sort | head -1 || true)
  [ -n "$tarball" ] || fail "images/watercast-*.tar.gz 가 없습니다"
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

log "5/6 기동"
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
  compose logs --tail 40 watercast >&2; fail "3분 안에 API 가 준비되지 않았습니다 (위 로그 참고)"; fi

log "6/6 확인"
if [ "$RUN_SMOKE" = "1" ]; then ./smoke_test.sh; fi
cat <<MSG

설치 완료
  API 문서   http://<이 서버>:${API_PORT}/docs
  업무 UI    http://<이 서버>:${UI_PORT}/
  데모 UI    http://<이 서버>:${PIPE_PORT}/   (COMPOSE_PROFILES=demo 일 때)
  상태       ./status.sh      중지 ./stop.sh      재기동 ./start.sh      제거 ./uninstall.sh
  연계 값    .env 의 BACKEND_LIAISON_* 를 채우고 ./start.sh
MSG
