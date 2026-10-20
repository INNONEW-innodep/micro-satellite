#!/usr/bin/env bash
# 컨테이너 상태와 API 헬스 (종료 코드: API 정상 0, 아니면 1)
. "$(dirname "$0")/common.sh"; load_env
compose ps -a
echo
if curl -fsS --max-time 5 "$API/api/v1/health" 2>/dev/null; then echo; ok "API 정상 ($API)"; else fail "API 응답 없음 ($API) — docker compose logs --tail 100 watercast"; fi
