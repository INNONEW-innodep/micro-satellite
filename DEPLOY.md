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

## API 문서(Swagger) 접속

| 문서 | 경로 |
|---|---|
| Swagger UI | `/docs` |
| ReDoc | `/redoc` |
| OpenAPI JSON | `/openapi.json` |

서버 공인 IP는 **211.243.12.176**(= semyeongsoft.com)이고 API는 호스트 18000이다.
다만 **현재 firewalld가 SSH(10322) 외 모든 포트를 막고 있어** 아래 주소는 포트를
개방해야 열린다. 개방은 서버 관리자만 할 수 있다(ssteam 계정에 sudo 없음).

```
개방 후:  http://211.243.12.176:18000/docs
```

개방 전에는 SSH 터널로 그대로 쓸 수 있다.

```bash
ssh -p 10322 -L 18000:localhost:18000 ssteam@semyeongsoft.com
# 브라우저에서 http://localhost:18000/docs
```

서버 안에서 확인만 할 때는 `curl -s http://localhost:18000/openapi.json`.

### 명세 문구를 고치려면

엔드포인트 설명은 라우트 데코레이터가 아니라 [`backend/app/api_docs.py`](backend/app/api_docs.py)에
모아 두었다. 라우트는 `description=API_DOCS["<키>"]`로 참조만 한다. 문구를 고칠 때
라우트 시그니처를 건드리지 않아도 되고, 어떤 엔드포인트에 설명이 빠졌는지 한 파일에서
바로 보인다.

- `SERVICE_DESCRIPTION` — Swagger 최상단 서비스 개요
- `TAG_DOCS` — `system` / `predictions` / `weather` / `wbms` 태그 그룹 설명
- `API_DOCS` — 엔드포인트 17개의 상세 설명(파라미터·단위·기본값·실패 응답·한계)
- `API_SUMMARIES` — 한국어 한 줄 요약(현재 라우트는 영문 summary를 쓰고 있어 미사용)

설명에는 기준선 모델·`rule_based` 위험도·조건부 `water_level_m` 같은 **한계 고지가
포함되어 있다.** 문구를 줄일 때 이 부분을 빼면 명세가 성능을 과장하게 되므로 유지한다.

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

### 서버에서 WBMS 실모델 GPU 추론 돌리기 (검증됨)

`wbms:0.13`은 서버에 적재돼 있고(11.25 GB), 부산 4시점 입력·가중치·마스크·DEM·GT가
`~/ssteam/watercast/data/`에 있다. 호스트에서 러너를 직접 실행하면 GPU로 돈다.

```bash
ssh -p 10322 ssteam@semyeongsoft.com
cd ~/ssteam/watercast
python3 scripts/wbms_chain_runner.py --scenes 20200302 --steps wb --gpu auto
# [gpu] 여유 VRAM 24077 MiB ≥ 3000 — GPU 사용
# [ok] wb_iceye_20200302 exit=0 38.0s
```

2026-08-20 기준 서버에는 **부산 4시점 실모델 산출이 모두 들어 있다**
(`data/wbms_runs/` — WB 마스크, 지점별 WLWA, LSTM 보정 수위).

| 씬 | Pre 크기 | GPU 시간 | IoU | Dice | Precision | Recall |
|---|---|---|---|---|---|---|
| 20200302 | 1.51 GB | 38.0초 | 0.9247 | 0.9609 | 0.9511 | 0.9709 |
| 20200330 | 1.14 GB | 28.7초 | 0.9472 | 0.9729 | 0.9608 | 0.9853 |
| 20200415 | 1.21 GB | 28.3초 | 0.9619 | 0.9806 | 0.9762 | 0.9850 |
| 20200416 | 1.59 GB | 34.8초 | 0.8810 | 0.9368 | 0.9474 | 0.9263 |

**평균 IoU 0.9287.** CPU 대비 13배(499초 → 38초)이며, 서버 4090은 전용이라 VRAM
프리플라이트가 항상 통과한다. 정확도는 배포 GT 라벨(`06_aux/Labels_GT`)과 직접 대조한
값이다. 20200416이 최저인데 배포 프로토콜 수치(IoU 0.8786, 20200416 시험씬 기준)와
거의 일치하므로 프로토콜이 재현된 것으로 본다.

전체 체인(수위·면적·융합)도 서버에서 돈다.

```bash
python3 scripts/wbms_chain_runner.py --steps wlwa,fused --gpu auto
# wlwa 4지점 각 0.6~0.7초, fused 4지점 각 2.5~2.6초, 합계 13초
```

서버에는 PlanetScope Pre가 없지만 **교차센서 쌍 부재는 WARN으로 처리되어 SAR 단독으로
융합 보정까지 산출된다**(exit=0 확인). 지점은 jeongcheon·hupo·gimhae·gupo 네 곳이고,
LSTM 번들은 이미지 동봉본이 자동 선택된다.

`--dry-run`으로 실행 없이 docker 명령만 확인할 수 있다.

### 체인 산출을 ui_next '정량 평가' 화면에 붙이기

`ui_next/model_eval.py`의 `load_batch_evals()`는 `data/eval/*.json`을 읽어 **'배치 산출 ·
체인 러너'** 계열로 표시한다. 배포 프로토콜·연구 프로토콜·스모크 재현과 섞이지 않는
별도 계열이므로, 체인 결과는 여기에만 넣는다. 기대 스키마는 아래로 충분하다.

```json
{ "wb": "WB_Busan_ICEYE_20200302T183857.tif",
  "metrics": { "water_iou": 0.924739, "f1": 0.960898 } }
```

파일명은 `WB_<테스트베드>_<센서>_<YYYYMMDD>T<HHMMSS>.tif` 규칙을 지켜야 센서·날짜가
파싱된다. 서버에는 4시점이 이미 생성돼 있고 로더가 `status: ok · 배치 산출 4건`으로
읽는 것을 확인했다.

이 경로는 **호스트 러너 전용**이다. 컨테이너 백엔드의 `/api/v1/wbms/status`는
`image_available: false`를 계속 보고하는데, 이미지가 없어서가 아니라 백엔드 컨테이너 안에
`docker` CLI가 없어서다(위 "남은 블로커" 참조).

### `wbms:0.13` 이미지를 다시 올려야 한다면

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
