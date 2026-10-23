#!/usr/bin/env bash
# WATERCAST 설치 패키지 빌드 — 다른 기관이 받아서 docker 로 올리는 묶음을 dist/ 에 만든다.
#
# 구성(2026-10-21): "런타임 컨테이너 + 앱 바이너리" 분리
#   images/watercast-runtime-<ver>.tar.gz      python 3.11 + 의존성 + docker CLI (앱 코드 없음) — 의존성이 바뀔 때만 교체
#   app/watercast-app-<ver>-<stamp>.tar.gz     앱 코드 묶음(backend·ui_next·pipeline_ui·regions·scripts·docker 진입점 + VERSION.json)
#                                              코드가 바뀌면 이것만 다시 만들어 ./update_app.sh 로 교체
#
#   scripts/package/build_release.sh                 기본 묶음 (런타임 이미지 + 앱 묶음 + 런타임 데이터 + 스크립트)
#   scripts/package/build_release.sh --app-only      앱 묶음만 dist/ 에 (코드 교체 배포용, 1분)
#   scripts/package/build_release.sh --with-wbms     + 실탐지 묶음 (wbms:0.13 이미지 11 GB, 모델·전처리·보조자료 28 GB)
#   scripts/package/build_release.sh --no-build      런타임 이미지를 다시 빌드하지 않고 기존 태그 사용
#
# 산출: dist/watercast-<ver>-<yyyymmdd>/ 와 dist/watercast-<ver>-<yyyymmdd>.tar.gz (실탐지 묶음은 별도 tar),
#       --app-only 는 dist/watercast-app-<ver>-<stamp>.tar.gz (+ .sha256)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$REPO"
VER=$(python3 -c "import re;print(re.search(r'version: str = \"([^\"]+)\"', open('backend/app/config.py').read()).group(1))")
STAMP=$(date +%Y%m%d)
BUILD_ID=$(date +%Y%m%d%H%M%S)
NAME="watercast-${VER}-${STAMP}"
OUT="dist/${NAME}"
IMAGE="watercast-runtime:${VER}"
GIT=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
WITH_WBMS=0; BUILD=1; APP_ONLY=0
for arg in "$@"; do case "$arg" in --with-wbms) WITH_WBMS=1;; --no-build) BUILD=0;; --app-only) APP_ONLY=1;; *) echo "알 수 없는 인자 $arg" >&2; exit 2;; esac; done
log() { printf '\033[1;34m[build]\033[0m %s\n' "$*"; }

# ── 앱 묶음 ────────────────────────────────────────────────────────────────
# .dockerignore 의 제외 규칙을 그대로 써서 예전 이미지에 들어가던 것과 같은 파일 집합을 담는다.
build_app_bundle() { # build_app_bundle <출력 디렉터리>
  local dest="$1" bundle="watercast-app-${VER}-${BUILD_ID}" stage
  stage=$(mktemp -d "${TMPDIR:-/tmp}/wc_app.XXXXXX")
  mkdir -p "$stage/$bundle"
  local excludes=()
  while IFS= read -r line; do
    line="${line%%#*}"; line="${line// /}"
    [ -z "$line" ] && continue
    case "$line" in
      '!'*) continue ;;
      '**/'*) excludes+=("--exclude=${line#**/}") ;;
      /*) excludes+=("--exclude=./${line#/}") ;;
      */*) excludes+=("--exclude=./$line") ;;
      *) excludes+=("--exclude=$line") ;;
    esac
  done < .dockerignore
  tar -c "${excludes[@]}" --exclude=./dist --exclude=./.git -f - . | tar -x -C "$stage/$bundle" -f -
  rm -rf "$stage/$bundle/backend/data" "$stage/$bundle/data"
  python3 - "$stage/$bundle/VERSION.json" "$VER" "$BUILD_ID" "$GIT" "$IMAGE" <<'PY'
import json, sys, datetime
path, ver, build, git, image = sys.argv[1:6]
json.dump({"version": ver, "build": build, "git": git, "runtime_image": image,
           "built_at": datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
           "note_ko": "설치 패키지 앱 묶음. 런타임 이미지(runtime_image)는 그대로 두고 ./update_app.sh 로 이 묶음만 교체한다."},
          open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
PY
  [ -f "$stage/$bundle/docker/entrypoint.sh" ] && [ -d "$stage/$bundle/backend/app" ] && [ -d "$stage/$bundle/regions" ] \
    || { echo "앱 묶음 내용이 비정상입니다(docker/entrypoint.sh·backend/app·regions 필요)" >&2; exit 1; }
  mkdir -p "$dest"
  tar -C "$stage" -czf "$dest/$bundle.tar.gz" "$bundle"
  (cd "$dest" && sha256sum "$bundle.tar.gz" > "$bundle.tar.gz.sha256")
  rm -rf "$stage"
  log "    앱 묶음 → $dest/$bundle.tar.gz ($(du -h "$dest/$bundle.tar.gz" | cut -f1), build $BUILD_ID, git $GIT)"
}

if [ "$APP_ONLY" = "1" ]; then
  log "앱 묶음만 빌드 (코드 교체 배포용)"
  build_app_bundle dist
  log "완료 — 서버에서: ./update_app.sh app/$(ls -t dist/watercast-app-*.tar.gz | head -1 | xargs basename)"
  exit 0
fi

log "패키지 $NAME → $OUT"
rm -rf "$OUT"; mkdir -p "$OUT"/{images,app,data,backend/data,src,logs}

# ── 1. 런타임 이미지 ───────────────────────────────────────────────────────
if [ "$BUILD" = "1" ]; then
  log "1/7 런타임 이미지 빌드 $IMAGE (Dockerfile --target runtime)"
  docker build --target runtime -t "$IMAGE" . >/dev/null
fi
docker image inspect "$IMAGE" >/dev/null
log "    docker save → images/watercast-runtime-${VER}.tar.gz"
docker save "$IMAGE" | gzip -1 > "$OUT/images/watercast-runtime-${VER}.tar.gz"

# ── 2. 앱 묶음 ─────────────────────────────────────────────────────────────
log "2/7 앱 묶음"
build_app_bundle "$OUT/app"

# ── 3. 런타임 데이터 (작음) — 탐지 캐시, 전처리 사이드카, 융합 번들, 기상 CSV ──
log "3/7 런타임 데이터"
H=data/incoming/handover
mkdir -p "$OUT/data/wbms_runs" "$OUT/data/incoming/handover"
rsync -a --include='*/' --include='WB_*' --exclude='*' data/wbms_runs/wb/ "$OUT/data/wbms_runs/wb/"
[ -d data/wbms_runs/wb_planet ] && rsync -a --include='*/' --include='WB_*' --exclude='*' data/wbms_runs/wb_planet/ "$OUT/data/wbms_runs/wb_planet/" || true
mkdir -p "$OUT/$H/05_l1_pre"
rsync -a --include='*/' --include='*_meta.json' --exclude='*' "$H/05_l1_pre/" "$OUT/$H/05_l1_pre/"
mkdir -p "$OUT/$H/02_package/deploy_package/config"
rsync -a "$H/02_package/deploy_package/config/" "$OUT/$H/02_package/deploy_package/config/"
mkdir -p "$OUT/$H/06_aux"
[ -f "$H/06_aux/busan_aws_2019_202004.csv" ] || { echo "융합 보정용 기상 CSV 가 없습니다: $H/06_aux/busan_aws_2019_202004.csv" >&2; exit 1; }
cp "$H/06_aux/busan_aws_2019_202004.csv" "$OUT/$H/06_aux/"
mkdir -p "$OUT"/backend/data/{regions,wbms_jobs,wbms_products,wbms_fusion,results}
touch "$OUT"/backend/data/.keep "$OUT"/logs/.keep

# ── 4. 소스 (호스트 러너용 — 표준 라이브러리만 쓴다) ────────────────────────
log "4/7 소스 (실탐지 러너·연계 모듈)"
mkdir -p "$OUT/src/scripts" "$OUT/src/backend/app/wbms"
cp scripts/wbms_chain_runner.py "$OUT/src/scripts/"
cp backend/app/wbms/liaison.py backend/app/wbms/liaison_log.py "$OUT/src/backend/app/wbms/"
touch "$OUT/src/backend/__init__.py" "$OUT/src/backend/app/__init__.py" "$OUT/src/backend/app/wbms/__init__.py"

# ── 5. 스크립트·설정·문서 ───────────────────────────────────────────
log "5/7 스크립트·설정·문서"
cp packaging/*.sh packaging/docker-compose.yml packaging/docker-compose.smss-net.yml packaging/.env.example "$OUT/"
sed -i "s|watercast-runtime:0.2.0|$IMAGE|g" "$OUT/docker-compose.yml" "$OUT/.env.example" "$OUT/common.sh"
cp packaging/README_설치.md "$OUT/README_설치.md"
cp packaging/배포_실행순서.md "$OUT/배포_실행순서.md"
cp packaging/패키지_안내.md "$OUT/패키지_안내.md"
cp docs/liaison/연계정리_이노뎁_2세부_작성본.md "$OUT/연계정리_이노뎁_2세부.md"
chmod +x "$OUT"/*.sh

# ── 6. 실탐지 묶음 (선택, 대용량) ────────────────────────────────────────────
if [ "$WITH_WBMS" = "1" ]; then
  log "6/7 실탐지 묶음 (wbms:0.13 + 모델·전처리·보조자료) — 수십 분"
  BIG="dist/${NAME}-detect"
  rm -rf "$BIG"; mkdir -p "$BIG/images" "$BIG/$H"
  docker save wbms:0.13 | gzip -1 > "$BIG/images/wbms-0.13.tar.gz"
  mkdir -p "$BIG/$H/03_model" "$BIG/$H/05_l1_pre" "$BIG/$H/06_aux"
  rsync -a "$H/03_model/" "$BIG/$H/03_model/"
  rsync -a "$H/05_l1_pre/" "$BIG/$H/05_l1_pre/"
  rsync -a --exclude='Labels_GT*' "$H/06_aux/" "$BIG/$H/06_aux/"
  cat > "$BIG/README_실탐지묶음.md" <<'MD'
# 실탐지 묶음
기본 패키지 디렉터리 **안으로** 푼다. 최상위 디렉터리를 벗겨야 images/ 와 data/ 가 합쳐진다.
    cd <기본 패키지 디렉터리>
    tar -xzf ../watercast-*-detect.tar.gz --strip-components=1
    sha256sum --quiet -c SHA256SUMS-detect      # 선택: 무결성 확인
그 다음  ./install.sh --with-wbms  로 wbms 이미지를 적재하고  ./detect_scene.sh <YYYYMMDD>  로 탐지한다.
MD
  (cd "$BIG" && find . -type f ! -name 'SHA256SUMS*' -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS-detect)
  tar -C dist -czf "dist/${NAME}-detect.tar.gz" "${NAME}-detect"
  log "    → dist/${NAME}-detect.tar.gz ($(du -h "dist/${NAME}-detect.tar.gz" | cut -f1))"
else
  log "6/7 실탐지 묶음 생략 (--with-wbms 로 생성)"
fi

# ── 7. 매니페스트·체크섬·tar ──────────────────────────────────────────
log "7/7 매니페스트·체크섬"
python3 - "$OUT" "$VER" "$IMAGE" "$STAMP" "$BUILD_ID" "$GIT" <<'PY'
import json, os, subprocess, sys
out, ver, image, stamp, build_id, git = sys.argv[1:7]
def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
files = []
for root, _, names in os.walk(out):
    for n in names:
        p = os.path.join(root, n); files.append((os.path.relpath(p, out), os.path.getsize(p)))
app = [f for f, _ in files if f.startswith("app/") and f.endswith(".tar.gz")]
manifest = {
    "name": os.path.basename(out), "version": ver, "built": stamp, "git": git,
    "runtime_image": image, "runtime_image_id": sh(f"docker image inspect -f '{{{{.Id}}}}' {image}"),
    "runtime_image_size_bytes": int(sh(f"docker image inspect -f '{{{{.Size}}}}' {image}") or 0),
    "app_bundle": app[0] if app else None, "app_build": build_id,
    "layout_ko": "런타임 컨테이너(images/) + 앱 바이너리(app/) 분리 — 코드만 바뀌면 app 묶음을 ./update_app.sh 로 교체, 의존성이 바뀌면 런타임 이미지 교체",
    "contents": {
        "images/": "런타임 이미지 watercast-runtime (python 3.11 + 의존성 + docker CLI, 앱 코드 없음)",
        "app/": "앱 묶음 watercast-app-<ver>-<build>.tar.gz (+ .sha256). install.sh 가 app/current 로 푼다",
        "data/wbms_runs/": "탐지 캐시 마스크 GeoTIFF + meta/qc (부산 ICEYE 4 · PlanetScope 4 · 대청 PlanetScope 10)",
        "data/incoming/handover/": "전처리 사이드카 JSON, 융합 LSTM 번들, 기상 CSV",
        "backend/data/": "쓰기 디렉터리 (잡 상태·규약 산출물·사용자 지역·예측 결과)",
        "src/": "호스트 실탐지 러너와 연계 모듈 (python3 표준 라이브러리만)",
        "logs/": "설치·교체·확인 스크립트 로그와 보고서",
    },
    "n_files": len(files), "total_bytes": sum(s for _, s in files),
    "requires": {"docker": ">=24", "docker compose": "v2", "python3": ">=3.9 (스모크·러너)", "ports": ["18000", "8501", "8502"]},
}
json.dump(manifest, open(os.path.join(out, "MANIFEST.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
PY
(cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
tar -C dist -czf "dist/${NAME}.tar.gz" "$NAME"
log "완료: dist/${NAME}.tar.gz ($(du -h "dist/${NAME}.tar.gz" | cut -f1))  디렉터리 $(du -sh "$OUT" | cut -f1)"
