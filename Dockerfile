# ── 1단계: docker CLI 정적 바이너리만 꺼낸다 ──────────────────────────────────────
# API 컨테이너가 호스트 /var/run/docker.sock 으로 wbms:0.13 을 직접 띄우려면 클라이언트
# 바이너리가 필요하다(데몬은 호스트 것). curl·tgz 는 이 단계에만 남고 런타임 이미지에는
# /usr/local/bin/docker 하나만 복사된다. 버전은 호스트 데몬과 같을 필요가 없다(API 버전 협상).
# https://download.docker.com/linux/static/stable/x86_64/ 에 있는 버전만 가능.
FROM python:3.11-slim AS docker-cli
ARG DOCKER_CLI_VERSION=28.3.3
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && curl -fsSL "https://download.docker.com/linux/static/stable/x86_64/docker-${DOCKER_CLI_VERSION}.tgz" \
       | tar -xz -C /usr/local/bin --strip-components=1 docker/docker \
    && /usr/local/bin/docker --version \
    && rm -rf /var/lib/apt/lists/*

# ── 2단계: 런타임 이미지 (의존성만, 앱 코드 없음) ─────────────────────────────────────
# 설치 패키지는 이 단계만 `docker build --target runtime` 으로 굽고, 앱 코드는 별도 묶음(app/watercast-app-*.tar.gz)을
# 호스트 ./app/current 에 풀어 /app 에 마운트한다 — 코드만 바뀌면 이미지는 그대로 두고 묶음만 교체한다(2026-10-21).
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# rasterio 휠은 GDAL을 번들하지만 libexpat 같은 시스템 라이브러리는 링크만 한다.
# python:3.11-slim 베이스에 libexpat1이 들어있지 않은 시점이 있으므로 명시적으로 설치한다.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libexpat1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
COPY backend/requirements.txt ./backend/requirements.txt
COPY ui_next/requirements.txt ./ui_next/requirements.txt
COPY scripts/requirements.txt ./scripts/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# docker CLI (클라이언트만). compose 가 /var/run/docker.sock 을 마운트하면 백엔드가 호스트
# 데몬으로 wbms 컨테이너를 띄운다(backend/app/wbms/runner.py). pip 레이어 뒤에 두어
# 의존성 캐시를 깨지 않는다.
COPY --from=docker-cli /usr/local/bin/docker /usr/local/bin/docker

# ── 3단계: 앱 코드를 구운 전체 이미지 (개발 compose `build: .` 의 기본 대상) ───────────────────
FROM runtime AS full

COPY . .

RUN chmod +x docker/*.sh

EXPOSE 8000 8501

ENTRYPOINT ["./docker/entrypoint.sh"]
