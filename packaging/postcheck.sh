#!/usr/bin/env bash
# 배포 후 확인 — 환경·설정·컨테이너·API·데이터 디렉터리·탐지 준비 상태를 한 번에 점검하고 보고서를 남긴다.
#
#   ./postcheck.sh            전부 점검 (스모크 테스트 18개 포함), 보고서 logs/postcheck_<시각>.txt
#   ./postcheck.sh --quick    스모크 테스트 생략(환경·컨테이너·헬스만)
#
# 종료 코드: 필수 항목이 전부 통과하면 0, 하나라도 실패하면 1 (선택 항목 실패는 "!" 로만 표시).
# 실패가 있으면 보고서 끝에 컨테이너 로그 200줄을 붙인다.
. "$(dirname "$0")/common.sh"; load_env
QUICK=0
for arg in "$@"; do case "$arg" in --quick) QUICK=1;; -h|--help) sed -n '2,8p' "$0"; exit 0;; *) fail "알 수 없는 인자: $arg";; esac; done
mkdir -p "$LOG_DIR"
REPORT="$LOG_DIR/postcheck_$(date +%Y%m%d_%H%M%S).txt"
PASS=0; FAILS=(); WARNS=()
req() { # req "이름" <명령...>  — 필수
  local name="$1"; shift
  if out=$("$@" 2>&1); then PASS=$((PASS+1)); ok "$name${out:+ — ${out%%$'\n'*}}"; printf 'OK   %s%s\n' "$name" "${out:+ — $out}" >>"$REPORT"
  else FAILS+=("$name"); printf '\033[1;31m  ✘\033[0m %s%s\n' "$name" "${out:+ — ${out%%$'\n'*}}"; printf 'FAIL %s\n%s\n' "$name" "$out" >>"$REPORT"; fi
}
opt() { # opt "이름" <명령...>  — 선택(실패해도 종료 코드에 안 들어감)
  local name="$1"; shift
  if out=$("$@" 2>&1); then PASS=$((PASS+1)); ok "$name${out:+ — ${out%%$'\n'*}}"; printf 'OK   %s%s\n' "$name" "${out:+ — $out}" >>"$REPORT"
  else WARNS+=("$name"); warn "$name${out:+ — ${out%%$'\n'*}}"; printf 'WARN %s\n%s\n' "$name" "$out" >>"$REPORT"; fi
}
sh_all() { bash -eo pipefail -c "$1"; }

{
  echo "WATERCAST 배포 후 확인  $(date '+%Y-%m-%d %H:%M:%S')  디렉터리 $HERE"
  echo "설정: API $API_PORT · UI $UI_PORT · 데모 $PIPE_PORT · 이미지 $IMAGE · 이름 $NAME · RUN_AS ${RUN_AS:-1000:1000} · WBMS_GPU ${WBMS_GPU:-auto} · KMA 키 $( [ -n "${KMA_API_KEY:-}" ] && echo 있음 || echo 없음 ) · 연계 URL $( [ -n "${BACKEND_LIAISON_LOG_URL:-}" ] && echo 있음 || echo 없음 ) · 프로필 ${COMPOSE_PROFILES:-}"
  echo "앱: $(app_version)"
  echo
} | tee "$REPORT"

log "A. 환경"
req "docker 데몬 접근" sh_all "docker info --format '{{.ServerVersion}}' | sed 's/^/docker /'"
req "docker compose v2" sh_all "docker compose version --short"
req "python3 (스크립트용)" sh_all "python3 --version"
req "디스크 여유 10 GB 이상 (이 디렉터리)" sh_all "a=\$(df -Pk . | awk 'NR==2{print \$4}'); echo \"\$((a/1024/1024)) GB 여유\"; [ \"\$a\" -ge 10485760 ]"
opt "nvidia-container-toolkit (GPU 탐지용, 없으면 CPU)" sh_all "command -v nvidia-ctk >/dev/null && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1"

log "B. 설정·디렉터리"
req ".env 있음" sh_all "[ -f .env ] && echo .env"
req "app/current 에 앱 묶음 (VERSION.json)" sh_all "[ -f app/current/VERSION.json ] && [ -f app/current/docker/entrypoint.sh ] && echo '$(app_version)'"
req "쓰기 디렉터리 backend/data" sh_all "for d in regions wbms_jobs wbms_products wbms_fusion results; do [ -d backend/data/\$d ] || exit 1; done; echo 5개"
req "탐지 캐시 data/wbms_runs/wb" sh_all "n=\$(find data/wbms_runs/wb -name 'WB_*_qc.json' 2>/dev/null | wc -l); echo \"\$n 장면\"; [ \"\$n\" -ge 1 ]"
opt "RUN_AS 가 data/wbms_runs 소유자와 같음 (탐지 산출 쓰기)" sh_all "o=\$(stat -c '%u:%g' data/wbms_runs); echo \"소유자 \$o · RUN_AS ${RUN_AS:-1000:1000}\"; [ \"\$o\" = '${RUN_AS:-1000:1000}' ]"
opt "새 장면 입력 위치 data/incoming/handover/05_l1_pre (실탐지 묶음)" sh_all "[ -d data/incoming/handover/05_l1_pre ] && find data/incoming/handover/05_l1_pre -name 'Pre_*.tif' | wc -l | sed 's/\$/ 장면 입력 파일/'"

log "C. 컨테이너·포트"
# 기동 직후에는 healthcheck 가 아직 'starting' 일 수 있어(간격 30초) 최대 90초 기다린다
wait_healthy() { # wait_healthy <컨테이너> → healthy 면 0
  local i s
  for ((i = 0; i < 18; i++)); do
    s=$(docker inspect -f '{{.State.Health.Status}}' "$1" 2>/dev/null || echo missing)
    if [ "$s" = healthy ]; then echo "healthy"; return 0; fi
    [ "$s" = starting ] || { echo "$s"; return 1; }
    sleep 5
  done
  echo "starting(90초 넘게)"; return 1
}
req "컨테이너 $NAME healthy" wait_healthy "$NAME"
case ",${COMPOSE_PROFILES:-}," in *,demo,*) req "컨테이너 $NAME-pipeline healthy (데모)" wait_healthy "$NAME-pipeline";; esac
req "컨테이너가 쓰는 이미지 = $IMAGE" sh_all "i=\$(docker inspect -f '{{.Config.Image}}' '$NAME'); echo \"\$i\"; [ \"\$i\" = '$IMAGE' ]"
req "API 포트 $API_PORT 응답" sh_all "curl -fsS --max-time 5 '$API/api/v1/health' >/dev/null && echo listening"
req "업무 UI 포트 $UI_PORT 응답" sh_all "curl -fsS --max-time 8 -o /dev/null -w '%{http_code}' 'http://127.0.0.1:$UI_PORT/' | grep -q 200 && echo 200"
opt "컨테이너 → 호스트 docker 소켓 (탐지 잡이 wbms 컨테이너를 띄우는 길)" sh_all "docker exec '$NAME' docker version --format 'server {{.Server.Version}}'"
opt "wbms:0.13 이미지 (실탐지용, 없으면 캐시 장면만)" sh_all "docker image inspect wbms:0.13 --format 'size {{.Size}}' | awk '{printf \"%.1f GB\n\", \$2/1e9}'"

log "D. API"
req "health status=ok · 버전" sh_all "curl -fsS '$API/api/v1/health' | python3 -c 'import json,sys; d=json.load(sys.stdin); assert d[\"status\"]==\"ok\"; b=d.get(\"build\") or {}; print(\"v\"+d[\"version\"], \"build\", b.get(\"build\",\"-\"), \"models\", d[\"models_loaded\"])'"
req "API 가 보고하는 build = app/current 의 build" sh_all "a=\$(curl -fsS '$API/api/v1/health' | python3 -c 'import json,sys; print((json.load(sys.stdin).get(\"build\") or {}).get(\"build\",\"\"))'); b=\$(python3 -c 'import json; print(json.load(open(\"app/current/VERSION.json\")).get(\"build\",\"\"))'); echo \"\$a\"; [ -n \"\$a\" ] && [ \"\$a\" = \"\$b\" ]"
req "지역 목록 (내장 지역 7)" sh_all "curl -fsS '$API/api/v1/regions' | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d[\"total\"], \"지역\"); assert d[\"total\"]>=1'"
req "수위 전망 API (구포대교 D+7, 인증 없이)" sh_all "curl -fsS '$API/api/v1/regions/nakdong_busan/gauge/%EA%B5%AC%ED%8F%AC%EB%8C%80%EA%B5%90/forecast?horizon=7' | python3 -c 'import json,sys; d=json.load(sys.stdin); print(len(d[\"steps\"]), \"일\", d[\"as_of\"])'"
req "오류 응답 봉투 (422 + message_ko + X-Request-ID)" sh_all "curl -sS -D - -o /tmp/wc_pc_e.json -X POST '$API/api/v1/regions/nakdong_busan/dam/collect' -H 'Content-Type: application/json' -d '{}' | grep -qi '^x-request-id:' && python3 -c 'import json; e=json.load(open(\"/tmp/wc_pc_e.json\"))[\"error\"]; assert e[\"status\"]==422; print(e[\"message_ko\"][:60])'"
req "최근 오류 조회 /errors/recent" sh_all "curl -fsS '$API/api/v1/errors/recent?limit=3' | python3 -c 'import json,sys; d=json.load(sys.stdin); print(\"누적\", d[\"counts\"])'"
req "간단 화면 /simple · 서비스 화면 /app · 문서 /docs" sh_all "for p in simple app docs; do curl -fsS -o /dev/null '$API/'\$p || exit 1; done; echo 3개"
opt "컨테이너 로그에 5xx 서버 오류 없음 (마지막 500줄)" sh_all "n=\$(docker logs --tail 500 '$NAME' 2>&1 | grep -c 'ERROR watercast.api' || true); echo \"서버 오류 \$n 건\"; [ \"\$n\" = 0 ]"

if [ "$QUICK" = "0" ]; then
  log "E. 스모크 테스트 (18개)"
  if ./smoke_test.sh >>"$REPORT" 2>&1; then PASS=$((PASS+1)); ok "스모크 테스트 통과 (상세는 보고서)"; else FAILS+=("스모크 테스트"); printf '\033[1;31m  ✘\033[0m 스모크 테스트 실패 — 보고서 참고\n'; fi
fi

{
  echo
  echo "결과: 통과 $PASS · 실패 ${#FAILS[@]} · 주의 ${#WARNS[@]}"
  [ ${#FAILS[@]} -gt 0 ] && printf '실패: %s\n' "${FAILS[@]}"
  [ ${#WARNS[@]} -gt 0 ] && printf '주의: %s\n' "${WARNS[@]}"
  true
} | tee -a "$REPORT"
if [ ${#FAILS[@]} -gt 0 ]; then
  { echo; echo "===== docker compose logs --tail 200 $NAME ====="; compose logs --tail 200 watercast 2>&1; } >>"$REPORT"
  printf '보고서(컨테이너 로그 포함): %s\n' "$REPORT"
  exit 1
fi
ok "배포 확인 완료 — 보고서 $REPORT"
