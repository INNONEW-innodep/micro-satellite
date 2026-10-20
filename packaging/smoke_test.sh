#!/usr/bin/env bash
# 설치 확인 — 공개 API 를 실제로 불러 18가지를 검사한다. 하나라도 실패하면 종료 코드 1.
# 3세부·통합 측이 받아 가는 기능(마스크 GeoTIFF 다운로드, 기간 필터, 규약 파일명, 로그 페이로드)이 들어 있다.
. "$(dirname "$0")/common.sh"; load_env
PASS=0; FAILS=()
check() { # check "이름" <명령> ; 명령이 0 이면 통과
  local name="$1"; shift
  if "$@" >/dev/null 2>&1; then PASS=$((PASS+1)); ok "$name"; else FAILS+=("$name"); printf '\033[1;31m  ✘\033[0m %s\n' "$name"; fi
}
# 여러 명령을 한 검사로 묶을 때 — 하나라도 실패하면 실패 (bash -c 는 기본으로 마지막 명령만 본다)
sh_all() { bash -eo pipefail -c "$1"; }
TMP=$(mktemp -d "${TMPDIR:-/tmp}/wc_smoke.XXXXXX"); trap 'rm -rf "$TMP"' EXIT

log "스모크 테스트: $API"
check "1. health status=ok" sh_all "curl -fsS '$API/api/v1/health' | grep -Eq '\"status\": *\"ok\"'"
check "2. 지역 목록에 nakdong_busan" sh_all "curl -fsS '$API/api/v1/regions' | grep -q nakdong_busan"
check "3. 위성 장면 8개" sh_all "curl -fsS '$API/api/v1/regions/nakdong_busan/satellite/scenes' | python3 -c 'import json,sys; d=json.load(sys.stdin); assert d[\"n_scenes\"]==8, d[\"n_scenes\"]'"
check "4. 장면 마스크 GeoTIFF 다운로드 (image/tiff, >1 MB)" sh_all "
  curl -fsS -o '$TMP/mask.tif' -w '%{content_type} %{size_download}' '$API/api/v1/regions/nakdong_busan/satellite/scenes/20200302_ICEYE/mask' | python3 -c 'import sys; ct,sz=sys.stdin.read().split(); assert ct.startswith(\"image/tiff\") and int(sz)>1_000_000, (ct,sz)'
  head -c 4 '$TMP/mask.tif' | grep -q 'II' "
check "5. 위성 수위 기간 필터 (2020-03, 구포대교 4건)" sh_all "curl -fsS '$API/api/v1/regions/nakdong_busan/satellite/levels?station_id=%EA%B5%AC%ED%8F%AC%EB%8C%80%EA%B5%90&start=2020-03-01&end=2020-03-31' | python3 -c 'import json,sys; d=json.load(sys.stdin); assert d[\"n\"]==4 and d[\"period\"][\"start\"]==\"2020-03-01\", d'"
check "6. 공통 격자 면적 시계열 available" sh_all "curl -fsS '$API/api/v1/regions/nakdong_busan/satellite/area-series' | grep -Eq '\"available\": *true'"
check "7. 수위 전망 D+7 (구포대교 2020-02-18)" sh_all "curl -fsS '$API/api/v1/regions/nakdong_busan/gauge/%EA%B5%AC%ED%8F%AC%EB%8C%80%EA%B5%90/forecast?as_of=2020-02-18&horizon=7' | python3 -c 'import json,sys; d=json.load(sys.stdin); assert len(d[\"steps\"])>=7'"
# 탐지 잡: 캐시된 장면이라 docker 없이 즉시 성공해야 하고, 규약 파일명 사본이 생겨야 한다.
# 응답 JSON 은 환경변수로 넘긴다 (파이썬 소스에 끼워 넣으면 따옴표·역슬래시가 든 응답에서 깨진다).
JOB_JSON=$(curl -fsS -X POST "$API/api/v1/wbms/segmentation/jobs" -H 'Content-Type: application/json' \
  -d '{"sensor":"iceye","testbed":"busan","image_date":"20200302","order_id":"smoke","scene_id":"1","work_id":"smoke-1"}' 2>/dev/null || echo '{}')
export JOB_JSON
JOB_ID=$(python3 -c 'import json,os; print(json.loads(os.environ["JOB_JSON"]).get("job_id",""))' 2>/dev/null || true)
check "8. 탐지 잡 캐시 히트 → succeeded + 연계 파라미터 보관" python3 -c "
import json,os; d=json.loads(os.environ['JOB_JSON']); assert d.get('status')=='succeeded' and d['request']['order_id']=='smoke', d"
check "9. 규약 파일명 사본 PT000003_{product_id}.tif 생성" python3 -c "
import json,os; d=json.loads(os.environ['JOB_JSON']); p=(d.get('result') or {}).get('product_tif') or ''; assert os.path.basename(p).startswith('PT000003_ICEYE_X5_SLC_SM_23467_20200302T183857'), p
assert os.path.isfile(os.path.join('backend','data','wbms_products', os.path.basename(p))), '호스트 backend/data/wbms_products 에 없음'"
check "10. 잡 마스크 다운로드 + 산출 파일 목록" sh_all "
  [ -n '$JOB_ID' ]
  curl -fsS -o /dev/null -w '%{http_code}' '$API/api/v1/wbms/segmentation/jobs/$JOB_ID/mask' | grep -q 200
  curl -fsS -o /dev/null -w '%{http_code}' '$API/api/v1/wbms/segmentation/jobs/$JOB_ID/files/WB_Busan_ICEYE_20200302T183857_qc.json' | grep -q 200"

check "11. 쉬운 화면 /easy 응답" bash -c "curl -fsS '$API/easy' | grep -q '위성 영상'"
check "12. 장면 파이프라인 옵션 (지역·센서 목록)" bash -c "curl -fsS '$API/api/v1/scenes/options' | python3 -c 'import json,sys; d=json.load(sys.stdin); assert any(r[\"id\"]==\"nakdong_busan\" for r in d[\"regions\"]) and {\"ICEYE\",\"PlanetScope\"} <= {s[\"id\"] for s in d[\"sensors\"]}'"

check "13. 서비스 화면 /app 응답" bash -c "curl -fsS '$API/app' | grep -q '네 단계'"
check "14. 모델 평가 화면 /eval · 준비 상태 API" bash -c "curl -fsS '$API/eval' | grep -q '성능' && curl -fsS '$API/api/v1/regions/nakdong_busan/readiness' | grep -q '\"gauge\"'"

# 2026-10-20 추가 — 오류 봉투·최근 오류 조회·시 단위 전망 API·간단 화면
check "15. 잘못된 입력 → 422 + error.message_ko(한국어)·X-Request-ID" sh_all "
  curl -sS -D '$TMP/h.txt' -o '$TMP/e.json' -X POST '$API/api/v1/regions/nakdong_busan/dam/collect' -H 'Content-Type: application/json' -d '{\"dam_code\":\"12\",\"start\":\"2024-13-01\"}'
  grep -qi '^x-request-id:' '$TMP/h.txt'
  python3 -c 'import json; d=json.load(open(\"$TMP/e.json\")); e=d[\"error\"]; assert e[\"status\"]==422 and \"잘못\" in e[\"message_ko\"] and e[\"fields\"][0][\"loc\"]==\"body.dam_code\", d'"
check "16. 최근 오류 조회 /errors/recent 에 방금 422 가 보임" sh_all "curl -fsS '$API/api/v1/errors/recent?limit=5' | python3 -c 'import json,sys; d=json.load(sys.stdin); assert d[\"n\"]>=1 and d[\"items\"][0][\"status\"]==422, d'"
check "17. 시 단위 자료 현황 API (지점 목록 응답)" sh_all "curl -fsS '$API/api/v1/regions/nakdong_busan/gauge/hourly' | python3 -c 'import json,sys; d=json.load(sys.stdin); assert \"stations\" in d and d[\"max_horizon_hours\"]==48, d'"
check "18. 간단 화면 /simple 응답 (단위 토글 포함)" bash -c "curl -fsS '$API/simple' | grep -q 'unitChips'"

echo
if [ ${#FAILS[@]} -eq 0 ]; then ok "스모크 테스트 ${PASS}/18 통과"; else
  printf '\033[1;31m실패 %d건:\033[0m\n' "${#FAILS[@]}"; printf '  - %s\n' "${FAILS[@]}"
  printf '  (자세히: ./status.sh, docker compose logs --tail 100 watercast)\n'; exit 1; fi
