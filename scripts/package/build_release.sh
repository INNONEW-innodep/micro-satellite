#!/usr/bin/env bash
# WATERCAST 설치 패키지 빌드 — 다른 기관이 받아서 docker 로 올리는 묶음을 dist/ 에 만든다.
#
#   scripts/package/build_release.sh                 기본 묶음 (이미지 + 런타임 데이터 + 스크립트)
#   scripts/package/build_release.sh --with-wbms     + 실탐지 묶음 (wbms:0.13 이미지 11 GB, 모델·전처리·보조자료 28 GB)
#   scripts/package/build_release.sh --no-build      이미지를 다시 빌드하지 않고 기존 태그 사용
#
# 산출: dist/watercast-<ver>-<yyyymmdd>/  와  dist/watercast-<ver>-<yyyymmdd>.tar.gz  (실탐지 묶음은 별도 tar)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$REPO"
VER=$(python3 -c "import re;print(re.search(r'version: str = \"([^\"]+)\"', open('backend/app/config.py').read()).group(1))")
STAMP=$(date +%Y%m%d)
NAME="watercast-${VER}-${STAMP}"
OUT="dist/${NAME}"
IMAGE="watercast:${VER}"
WITH_WBMS=0; BUILD=1
for arg in "$@"; do case "$arg" in --with-wbms) WITH_WBMS=1;; --no-build) BUILD=0;; *) echo "알 수 없는 인자 $arg" >&2; exit 2;; esac; done
log() { printf '\033[1;34m[build]\033[0m %s\n' "$*"; }

log "패키지 $NAME → $OUT"
rm -rf "$OUT"; mkdir -p "$OUT"/{images,data,backend/data,src}

# ── 1. 이미지 ─────────────────────────────────────────────────────────────
if [ "$BUILD" = "1" ]; then
  log "1/6 이미지 빌드 $IMAGE"
  docker build -t "$IMAGE" -t watercast:latest . >/dev/null
fi
docker image inspect "$IMAGE" >/dev/null
log "    docker save → images/watercast-${VER}.tar.gz"
docker save "$IMAGE" | gzip -1 > "$OUT/images/watercast-${VER}.tar.gz"

# ── 2. 런타임 데이터 (작음) — 탐지 캐시, 전처리 사이드카, 융합 번들, 기상 CSV ──
log "2/6 런타임 데이터"
H=data/incoming/handover
mkdir -p "$OUT/data/wbms_runs" "$OUT/data/incoming/handover"
# 캐시된 탐지 마스크(8씬 GeoTIFF + meta + qc) — 캐시 히트·규약 파일명·마스크 다운로드에 필요
rsync -a --include='*/' --include='WB_*' --exclude='*' data/wbms_runs/wb/ "$OUT/data/wbms_runs/wb/"
[ -d data/wbms_runs/wb_planet ] && rsync -a --include='*/' --include='WB_*' --exclude='*' data/wbms_runs/wb_planet/ "$OUT/data/wbms_runs/wb_planet/" || true
# 전처리 사이드카(JSON 만) — product_id 파생용
mkdir -p "$OUT/$H/05_l1_pre"
rsync -a --include='*/' --include='*_meta.json' --exclude='*' "$H/05_l1_pre/" "$OUT/$H/05_l1_pre/"
# 융합 LSTM 번들·기상 CSV·카탈로그 보조 — 융합 보정 API 용
mkdir -p "$OUT/$H/02_package/deploy_package/config"
rsync -a "$H/02_package/deploy_package/config/" "$OUT/$H/02_package/deploy_package/config/"
mkdir -p "$OUT/$H/06_aux"
[ -f "$H/06_aux/busan_aws_2019_20204.csv" ] || [ -f "$H/06_aux/busan_aws_2019_202004.csv" ] || { echo "융합 보정용 기상 CSV 가 없습니다: $H/06_aux/busan_aws_2019_202004.csv" >&2; exit 1; }
cp "$H/06_aux/busan_aws_2019_202004.csv" "$OUT/$H/06_aux/"
# 쓰기 디렉터리 골격 (설치 시 install.sh 가 다시 만든다)
mkdir -p "$OUT"/backend/data/{regions,wbms_jobs,wbms_products,wbms_fusion,results}
touch "$OUT"/backend/data/.keep

# ── 3. 소스 (호스트 러너용 — 표준 라이브러리만 쓴다) ────────────────────────
log "3/6 소스 (실탐지 러너·연계 모듈)"
mkdir -p "$OUT/src/scripts" "$OUT/src/backend/app/wbms"
cp scripts/wbms_chain_runner.py "$OUT/src/scripts/"
cp backend/app/wbms/liaison.py backend/app/wbms/liaison_log.py "$OUT/src/backend/app/wbms/"
touch "$OUT/src/backend/__init__.py" "$OUT/src/backend/app/__init__.py" "$OUT/src/backend/app/wbms/__init__.py"

# ── 4. 스크립트·설정·문서 ─────────────────────────────────────────────────
log "4/6 스크립트·설정·문서"
cp packaging/*.sh packaging/docker-compose.yml packaging/docker-compose.smss-net.yml packaging/.env.example "$OUT/"
sed -i "s|watercast:0.2.0|$IMAGE|g" "$OUT/docker-compose.yml" "$OUT/.env.example" "$OUT/common.sh"
cp packaging/README_설치.md "$OUT/README_설치.md"
cp docs/liaison/연계정리_이노뎁_2세부_작성본.md "$OUT/연계정리_이노뎁_2세부.md"
chmod +x "$OUT"/*.sh

# ── 5. 실탐지 묶음 (선택, 대용량) ─────────────────────────────────────────
if [ "$WITH_WBMS" = "1" ]; then
  log "5/6 실탐지 묶음 (wbms:0.13 + 모델·전처리·보조자료) — 수십 분"
  BIG="dist/${NAME}-detect"
  rm -rf "$BIG"; mkdir -p "$BIG/images" "$BIG/$H"
  docker save wbms:0.13 | gzip -1 > "$BIG/images/wbms-0.13.tar.gz"
  mkdir -p "$BIG/$H/03_model" "$BIG/$H/05_l1_pre" "$BIG/$H/06_aux"
  rsync -a "$H/03_model/" "$BIG/$H/03_model/"
  rsync -a "$H/05_l1_pre/" "$BIG/$H/05_l1_pre/"
  # 정답 라벨(학습·검증용 2.2 GB)은 탐지에 쓰지 않으므로 뺀다
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
  log "5/6 실탐지 묶음 생략 (--with-wbms 로 생성)"
fi

# ── 6. 매니페스트·체크섬·tar ───────────────────────────────────────────────
log "6/6 매니페스트·체크섬"
python3 - "$OUT" "$VER" "$IMAGE" "$STAMP" <<'PY'
import json, os, subprocess, sys
out, ver, image, stamp = sys.argv[1:5]
def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
files = []
for root, _, names in os.walk(out):
    for n in names:
        p = os.path.join(root, n); files.append((os.path.relpath(p, out), os.path.getsize(p)))
manifest = {
    "name": os.path.basename(out), "version": ver, "image": image, "built": stamp,
    "git": sh("git rev-parse --short HEAD 2>/dev/null"), "image_id": sh(f"docker image inspect -f '{{{{.Id}}}}' {image}"),
    "image_size_bytes": int(sh(f"docker image inspect -f '{{{{.Size}}}}' {image}") or 0),
    "contents": {
        "images/": "watercast API+UI 이미지 (docker load)",
        "data/wbms_runs/": "탐지 캐시 마스크 GeoTIFF + meta/qc (부산 ICEYE 4 · PlanetScope 4 · 대청 PlanetScope 10)",
        "data/incoming/handover/": "전처리 사이드카 JSON, 융합 LSTM 번들, 기상 CSV",
        "backend/data/": "쓰기 디렉터리 (잡 상태·규약 산출물·사용자 지역)",
        "src/": "호스트 실탐지 러너와 연계 모듈 (python3 표준 라이브러리만)",
    },
    "n_files": len(files), "total_bytes": sum(s for _, s in files),
    "requires": {"docker": ">=24", "docker compose": "v2", "python3": ">=3.9 (스모크·러너)", "ports": ["18000", "8501", "8502"]},
}
json.dump(manifest, open(os.path.join(out, "MANIFEST.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
PY
(cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
tar -C dist -czf "dist/${NAME}.tar.gz" "$NAME"
log "완료: dist/${NAME}.tar.gz ($(du -h "dist/${NAME}.tar.gz" | cut -f1))  디렉터리 $(du -sh "$OUT" | cut -f1)"
