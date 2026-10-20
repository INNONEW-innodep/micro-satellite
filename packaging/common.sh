# 설치 스크립트 공통 — 모든 스크립트가 source 한다. 패키지 디렉터리에서 실행한다.
# bash + coreutils + curl + python3(표준 라이브러리) + docker compose v2 만 쓴다.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

log()  { printf '\033[1;34m[watercast]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m  ✔\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m  !\033[0m %s\n' "$*" >&2; }
fail() { printf '\033[1;31m  ✘ %s\033[0m\n' "$*" >&2; exit 1; }

need() { command -v "$1" >/dev/null 2>&1 || fail "$1 이 필요합니다: $2"; }

load_env() {
  # .env 의 KEY=VALUE 를 셸 변수로. compose 와 같은 규칙으로 읽는다:
  #   주석·빈 줄 무시, CRLF(윈도 편집기) 허용, "값"·'값' 따옴표 벗김, 값 안의 = 는 그대로,
  #   따옴표 없는 값 뒤의 " # 주석" 제거. 명령줄·환경으로 이미 준 값이 우선한다.
  local line key value dq="^\"(.*)\"[[:space:]]*(#.*)?$" sq="^'(.*)'[[:space:]]*(#.*)?$"
  if [ -f .env ]; then
    while IFS= read -r line || [ -n "$line" ]; do
      line="${line%$'\r'}"
      [[ "$line" =~ ^[[:space:]]*(#|$) ]] && continue
      [[ "$line" == *=* ]] || continue
      key="${line%%=*}"; value="${line#*=}"
      key="${key//[[:space:]]/}"
      [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
      value="${value#"${value%%[![:space:]]*}"}"
      if [[ "$value" =~ $dq ]] || [[ "$value" =~ $sq ]]; then
        value="${BASH_REMATCH[1]}"
      else
        value="${value%%[[:space:]]#*}"
        value="${value%"${value##*[![:space:]]}"}"
      fi
      if [ -z "${!key+x}" ]; then export "$key=$value"; fi
    done < .env
  fi
  API_PORT="${WATERCAST_API_PORT:-18000}"
  UI_PORT="${WATERCAST_UI_PORT:-8501}"
  PIPE_PORT="${WATERCAST_PIPE_PORT:-8502}"
  IMAGE="${WATERCAST_IMAGE:-watercast:0.2.0}"
  NAME="${WATERCAST_NAME:-watercast}"
  API="http://127.0.0.1:${API_PORT}"
  : "$UI_PORT" "$PIPE_PORT" "$IMAGE" "$NAME" "$API"
}

compose() {
  # 공유 네트워크 오버레이는 WITH_SMSS_NET=1 (.env 또는 환경) 일 때만 덧붙인다
  local files=(-f docker-compose.yml)
  if [ "${WITH_SMSS_NET:-0}" = "1" ]; then files+=(-f docker-compose.smss-net.yml); fi
  docker compose "${files[@]}" "$@"
}

compose_up() {
  # 데모 프로필(COMPOSE_PROFILES)을 껐을 때 이전에 떠 있던 데모 컨테이너를 먼저 정리한다
  # (--remove-orphans 는 프로필로 꺼진 서비스를 고아로 보지 않아 그대로 남는다)
  case ",${COMPOSE_PROFILES:-}," in
    *,demo,*) ;;
    *) COMPOSE_PROFILES=demo compose rm -sf pipeline-ui >/dev/null 2>&1 || true ;;
  esac
  compose up -d --remove-orphans
}

compose_down() {
  # 프로필(demo)을 꺼 둔 뒤 내려도 데모 컨테이너까지 같이 내려가도록 모든 프로필을 켜고 내린다
  COMPOSE_PROFILES=demo compose down --remove-orphans
}

container_running() { # container_running <이름>
  [ -n "$(docker ps -q --filter "name=^/?$1\$" 2>/dev/null)" ]
}

port_in_use() {
  # 호스트에서 TCP <포트> 를 LISTEN 중이면 0. ss/lsof 없이 /proc 만 읽는다 (최소 설치 Rocky 에도 동작)
  python3 - "$1" <<'PY'
import sys
port = int(sys.argv[1]); hit = False
for path in ("/proc/net/tcp", "/proc/net/tcp6"):
    try:
        with open(path) as fh:
            next(fh)
            for ln in fh:
                f = ln.split()
                if len(f) > 3 and f[3] == "0A" and int(f[1].rsplit(":", 1)[1], 16) == port:
                    hit = True
    except (OSError, ValueError, StopIteration):
        pass
sys.exit(0 if hit else 1)
PY
}

wait_health() { # wait_health <url> [시도 횟수] — 2초 간격
  local url="$1" tries="${2:-90}" i
  for ((i = 0; i < tries; i++)); do
    if curl -fsS --max-time 5 "$url" 2>/dev/null | grep -Eq '"status": *"ok"'; then return 0; fi
    sleep 2
  done
  return 1
}
