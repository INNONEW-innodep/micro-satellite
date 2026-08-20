FROM python:3.11-slim

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

COPY . .

RUN chmod +x docker/entrypoint.sh

EXPOSE 8000 8501

ENTRYPOINT ["./docker/entrypoint.sh"]
