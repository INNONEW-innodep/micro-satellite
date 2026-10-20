#!/usr/bin/env bash
# 기동·재기동 (.env 변경 반영). WITH_SMSS_NET=1 (.env 또는 환경) 이면 공유 네트워크 오버레이 포함.
. "$(dirname "$0")/common.sh"; load_env
# 데모 프로필을 끈 뒤 재기동하면 데모 컨테이너는 내린다 (compose 는 프로필 밖 서비스를 고아로 보지 않는다)
case ",${COMPOSE_PROFILES:-}," in
  *,demo,*) ;;
  *) if container_running "$NAME-pipeline"; then COMPOSE_PROFILES=demo compose rm -sf pipeline-ui >/dev/null 2>&1 || true; fi ;;
esac
compose up -d --remove-orphans
if wait_health "$API/api/v1/health" 90; then ok "API 준비: $API/docs"
else compose logs --tail 40 watercast >&2; fail "3분 안에 API 가 준비되지 않았습니다 (위 로그 참고)"; fi
