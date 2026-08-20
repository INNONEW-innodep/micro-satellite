# semyeongsoft 서버 배포 (WATERCAST)

## 배포 위치
- 서버: `ssh -p 10322 ssteam@semyeongsoft.com` (비밀번호는 전달 메일 참조)
- 경로: `/home/ssteam/ssteam/watercast/`
- 컨테이너 2개 (docker compose, 둘 다 `restart: unless-stopped`)
  - `watercast` — FastAPI + 업무 UI(`ui_next`)
  - `watercast-pipeline` — 발표용 파이프라인 데모(`pipeline_ui`). 같은 이미지에 진입점만 다르다.

## 포트
| 용도 | 컨테이너 | 호스트 | 비고 |
|---|---|---|---|
| 업무 UI (`ui_next`) | 8501 | 8501 | |
| 파이프라인 데모 (`pipeline_ui`) | 8501 | **8502** | |
| FastAPI | 8000 | **18000** | 호스트 8000은 기존 `sat-water-seg-api`가 점유 |

외부에서는 SSH(10322) 외 모든 포트가 방화벽으로 차단되어 있다. 외부 공개가 필요하면
전달 메일 안내대로 서버 관리자에게 8501·8502·18000 개방을 요청한다. 개방 전에는 SSH 터널로
확인할 수 있다.

```bash
ssh -p 10322 -L 8501:localhost:8501 -L 8502:localhost:8502 \
    -L 18000:localhost:18000 ssteam@semyeongsoft.com
# 업무 UI      http://localhost:8501
# 파이프라인 데모 http://localhost:8502
# API 문서      http://localhost:18000/docs
```

## 서버에서 운영
```bash
cd ~/ssteam/watercast
docker compose logs -f          # 로그
docker compose restart          # 재시작
docker compose up -d --build    # 코드 변경 반영
```

## 기상청 ASOS 키

`~/ssteam/watercast/.env`에 `KMA_API_KEY=<일반 인증키>` 한 줄로 설정되어 있고(권한 600),
`docker compose up -d`가 이를 컨테이너 환경변수로 넘긴다. 설정 상태는 아래로 확인한다.

```bash
curl -s http://localhost:18000/api/v1/weather/status   # asos_available: true 이면 실조회 가능
```

키를 바꿀 때는 `.env`를 수정하고 `docker compose up -d`로 재생성한다(`restart`만으로는 반영되지 않음).
키는 Encoding/Decoding 어느 형식이든 되는데, 클라이언트가 `unquote` 후 다시 인코딩하기 때문이다
(`weather_service/client.py`). 키가 없으면 UI/API는 sample 모드로 동작한다.
`.env`는 저장소에 커밋하지 않는다(.gitignore 처리됨).

## 로컬에서 재배포
로컬 변경분을 tar로 만들어 올리고 서버에서 다시 빌드한다. (서버 `authorized_keys`는
건드리지 않는 방침이라 비밀번호 인증 기준)

```bash
tar --exclude='.git' --exclude='.venv' --exclude='*/.venv' \
    --exclude='__pycache__' --exclude='*/__pycache__' \
    --exclude='.pytest_cache' --exclude='.ruff_cache' \
    --exclude='*.pptx' --exclude='.claude' --exclude='backend/data' \
    --exclude='./data' --exclude='*.h5' \
    -czf /tmp/watercast_code.tar.gz .
scp -P 10322 /tmp/watercast_code.tar.gz ssteam@semyeongsoft.com:~/ssteam/watercast/
ssh -p 10322 ssteam@semyeongsoft.com \
  'cd ~/ssteam/watercast && cp .env /tmp/env.bak && tar -xzf watercast_code.tar.gz \
   && rm watercast_code.tar.gz && cp /tmp/env.bak .env && chmod 600 .env \
   && chmod +x docker/*.sh && docker compose up -d --build'
```

`--exclude='./data' --exclude='*.h5'`가 중요하다. UI 런타임은 `data/`를 읽지 않는다.
두 UI는 `ui_next/assets/`와 `pipeline_ui/assets/`의 파생 자산만 읽으므로 코드 tar에는
이것만 담고, 대용량 원본은 아래처럼 따로 올린다.
(같은 이유로 `.dockerignore`에도 넣어 빌드 컨텍스트가 커지지 않게 했다.)

## 원본 SAR 데이터 (`data/`, 3.7 GB)

`~/ssteam/watercast/data/`에 올려두고 두 컨테이너에 `/app/data`로 **읽기전용 마운트**한다
(docker-compose.yml). 이미지에는 굽지 않는다. 코드 재배포용 tar와 별개이므로 위 tar 절차는
`data/`를 건드리지 않는다 — 한 번 올리면 유지된다.

```bash
rsync -av -e 'ssh -p 10322' data/ ssteam@semyeongsoft.com:~/ssteam/watercast/data/
```

들어있는 부산 2020-03-02 ICEYE 원본:

| 파일 | 크기 | 내용 |
|---|---|---|
| `nas_staging/20200302_Input.tif` | 1.46 GB | 19571×24857, EPSG:32652, 3 m, float32 |
| `nas_staging/20200302_Label.tif` | 489 MB | 같은 격자 수체 라벨 (0/1) |
| `ingest/Processed_20200302_ICEYE.tif` | 1.13 GB | 지오코딩 처리본 |
| `ingest/20200302_ICEYE_sigma0db_slant.tif` | 550 MB | slant-range sigma0 dB (CRS 없음) |
| `ingest/ICEYE_ai_data_20200302.pkl` | 263 MB | 학습용 패치 201쌍 |
| `audit/`, `ingest/*.json` | 소량 | sigma0 전 장면 통계, 지오코딩 리포트 |

원본에서 실측치를 직접 확인하려면:

```bash
docker exec watercast python -c "
import rasterio, numpy as np
with rasterio.open('/app/data/nas_staging/20200302_Label.tif') as ds:
    a = ds.read(1); px = abs(ds.transform.a)*abs(ds.transform.e)
    print('수체', int((a==1).sum()), '픽셀 ->', (a==1).sum()*px/1e6, 'km2')"
# -> 4,884,814 픽셀 -> 43.963 km2 (전 장면)
```

이 값은 UI가 표시하는 43.673 km²(4시점 공통영역 교집합)와 0.66% 차이로 일치한다. 범위가
다를 뿐 둘 다 같은 원본에서 나온 실측치다.

tar가 `.env`를 덮어쓰지 않도록 위 명령처럼 백업·복원한다.

`backend/data/`(예측 결과 저장소)는 호스트 바인드 마운트라 재배포해도 유지된다.
컨테이너가 root로 쓰기 때문에 결과 파일을 호스트에서 지울 때는
`docker exec watercast rm -rf /app/backend/data/results/<id>` 형태로 지운다.

## GPU (2026-08-20 확인)

서버는 GPU 컨테이너를 **이미 돌릴 수 있는 상태**다. 추가 설치가 필요 없다.

| 항목 | 상태 |
|---|---|
| GPU | RTX 4090 24 GB, 유휴 (5 MiB 사용) |
| 드라이버 | 575.57.08 (CUDA 12.9) |
| nvidia-container-toolkit | 1.17.8 설치됨 |
| docker `nvidia` 런타임 | `/etc/docker/daemon.json`에 등록됨 |
| 실측 확인 | `docker run --rm --gpus all --entrypoint nvidia-smi <image> -L` → GPU 0 인식 |

현재 `watercast`/`watercast-pipeline`은 CPU 전용(`python:3.11-slim`, torch 없음)이고
GPU 예약도 없다. 이 앱들은 GPU가 필요 없으므로 그대로 두는 것이 맞다.

### WBMS 실모델 — 서버에 올라간 것과 남은 블로커

`~/ssteam/watercast/data/`에 WBMS 체인 입력 일체를 올려두었다(총 8.9 GB). 46 GB 전량이
아니라 실제로 마운트되는 것만 골랐다 — `06_aux/Masks`(6.4 GB), `DEM`(3.5 GB) 등은
런타임이 읽지 않는다.

| 경로 | 크기 | 용도 |
|---|---|---|
| `incoming/handover/03_model/` | 118 MB | `WBMS_SAR_ICEYE.h5` 등 가중치 |
| `incoming/handover/02_package/` | 5.7 MB | LSTM 융합 번들 설정 |
| `incoming/handover/05_l1_pre/iceye_pre/` | 5.1 GB | 부산 4시점 전처리 입력 |
| `incoming/handover/06_aux/*.csv` | 10 KB | AWS 기상, WAMIS 수위 |
| `incoming/handover/07_test_evidence/`, `wb_smoke/` | 4 MB | 정량 평가 근거 |

결과: `GET /api/v1/wbms/status`의 `handover_available`·`bundle_available`·
`aws_csv_available`가 모두 true가 되고, `ui_next/tests/test_model_eval.py` 16개가
서버에서 전부 통과한다(이전에는 9개 스킵).

**남은 블로커는 데이터가 아니라 구조다.** 백엔드는 컨테이너 안에서 도는데 그 안에
`docker` CLI가 없다. `runner.py`는 호스트 `docker`로 shell out 하는 설계이고 주석에
docker.sock 마운트를 보안상 하지 않는다고 못박혀 있다. 그래서 `image_available`은
`wbms:0.13`을 서버에 적재해도 계속 false다 — 이미지가 없어서가 아니라 백엔드가
물어볼 수단이 없어서다. 같은 이유로 이미지 11 GB는 **일부러 올리지 않았다**(올려도
관측 가능한 변화가 없다).

라이브 체인을 서버에서 돌리려면 둘 중 하나를 골라야 한다.
1. 백엔드를 호스트에서 직접 실행(컨테이너 밖). 현재 배포 형태를 바꾸는 결정이다.
2. 호스트측 러너를 따로 두고 백엔드는 작업 요청만 파일로 남긴다. 새 구성요소가 필요하다.

`docker.sock` 마운트는 컨테이너에 호스트 root 권한을 주는 것과 같아 선택지에서 뺐다.

**시연에는 영향이 없다.** `pipeline_ui`는 미리 생성한 실모델 산출 자산
(`assets/wbms_*.png` + `wbms_20200302_meta.json`)을 읽으므로 라이브 체인 없이도
실모델 결과를 보여준다.

### 전달받은 `wbms:0.13` 이미지를 서버에 올리려면

로컬 `wbms_image_0.13_20260818-001.tar` (11 GB)는 CUDA 11.8 기반 GPU 이미지다
(`NVIDIA_VISIBLE_DEVICES=all`, driver≥450 요구, `WORKDIR /WBMS`, 포트 1223,
`CMD ["sleep","infinity"]` — exec으로 들어가 쓰는 형태). 서버 드라이버 575가 요구치를
넘으므로 호환된다.

```bash
scp -P 10322 wbms_image_0.13_20260818-001.tar ssteam@semyeongsoft.com:~/ssteam/
ssh -p 10322 ssteam@semyeongsoft.com \
  'cd ~/ssteam && docker load -i wbms_image_0.13_20260818-001.tar && rm -f wbms_image_0.13_20260818-001.tar'
# 실행 예 (GPU 붙여서)
docker run -d --name wbms --gpus all -p 1223:1223 wbms:0.13
```

전송 약 12분(15 MB/s), 디스크는 tar 11 GB + 적재 이미지가 필요하다. 서버 여유가
101 GB이므로 가능하지만 `docker load` 직후 tar는 지운다. compose로 붙일 경우 GPU는
`deploy.resources.reservations.devices`로 예약한다.

## 배포 후 검증

```bash
# 1) 두 컨테이너가 healthy 인가
docker ps --filter name=watercast --format "table {{.Names}}\t{{.Status}}"

# 2) API·UI 응답
curl -s http://localhost:18000/api/v1/health
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8501/
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8502/

# 3) 전체 테스트 (운영 컨테이너를 건드리지 않는 일회용 컨테이너)
docker run --rm --entrypoint sh watercast:latest -c \
  "pip install -q pytest httpx2; python -m pytest ui_next/tests backend/tests weather_service/tests -q"
```

`pytest`와 `httpx2`는 테스트 전용이라 이미지에 넣지 않았다. 그래서 운영 컨테이너에서
`unittest discover`를 돌리면 `ModuleNotFoundError: pytest`로 8건이 실패하는데, 이는
정상이며 위 일회용 컨테이너 방식으로 확인한다.
