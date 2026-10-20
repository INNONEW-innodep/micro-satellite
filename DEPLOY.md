# semyeongsoft 서버 배포 (WATERCAST)

## 배포 위치
- 서버: `ssh -p 10322 ssteam@semyeongsoft.com` (비밀번호는 전달 메일 참조)
- 경로: `/home/ssteam/ssteam/watercast/`
- 컨테이너 2개 (docker compose, 둘 다 `restart: unless-stopped`)
  - `watercast` — FastAPI + 업무 UI(`ui_next`)
  - `watercast-pipeline` — 발표용 파이프라인 데모(`pipeline_ui`). 같은 이미지에 진입점만 다르다.

## 포트와 외부 주소
| 용도 | 컨테이너 | 호스트 | 외부 HTTPS (프록시) |
|---|---|---|---|
| 업무 UI (`ui_next`) | 8501 | 8501 | https://wc-ui.semyeongsoft.com |
| 파이프라인 데모 (`pipeline_ui`) | 8501 | **8502** | https://wc-pipe.semyeongsoft.com |
| FastAPI | 8000 | **18000** | https://wc-api.semyeongsoft.com (`/docs`) |

호스트 8000은 기존 `sat-water-seg-api`가 점유하고 있어 18000으로 매핑했다.

> **2026-09-08 실측 — 세 도메인은 인터넷에 공개되어 있고 인증이 없다.** 호스트 포트
> 18000·8501·8502는 방화벽에 막혀 있지만 서버 관리자가 둔 HTTPS 프록시(Let's Encrypt,
> 인증서 만료 2026-11-09)가 그 앞에서 열려 있다. API에는 인증·요청 제한이 없고 CORS는
> 모든 오리진을 허용하므로, 누구나 서버 KMA 키로 ASOS 조회를 하거나 예측을 생성·열람할
> 수 있다. 프록시 단 Basic Auth/IP allowlist 또는 `X-API-Key` 의존성 추가가 필요하며,
> 이는 관리자·사용자 결정 사항이다. (이전 문서의 "SSH 외 전부 차단"은 호스트 포트에만 해당한다.)

프록시 설정 파일은 이 저장소에 없다(서버 관리자 소관). 개발 중 SSH 터널로도 볼 수 있다.

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

외부: https://wc-api.semyeongsoft.com/docs · 서버 안: `curl -s http://localhost:18000/openapi.json`.

태그: `system` · `predictions`(마스크 시계열) · `weather` · `wbms`(인도 실모델 실행,
`/segmenters` 포함) · **`regions`(지역 레지스트리 + 지상 수위계 시계열 예측, 2026-09-08 추가)**.

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
`.env`는 저장소에 커밋하지 않고(.gitignore) **이미지에도 넣지 않는다**(.dockerignore, 2026-09-08부터).
그 전에 빌드한 이미지에는 `/app/.env`가 레이어로 남아 있으므로 키를 바꾼 뒤 `docker compose build --no-cache`로 다시 굽는다.

## 로컬에서 재배포
로컬 변경분을 tar로 만들어 올리고 서버에서 다시 빌드한다. (서버 `authorized_keys`는
건드리지 않는 방침이라 비밀번호 인증 기준)

```bash
tar --exclude='.git' --exclude='.venv' --exclude='*/.venv' \
    --exclude='__pycache__' --exclude='*/__pycache__' \
    --exclude='.pytest_cache' --exclude='.ruff_cache' \
    --exclude='*.pptx' --exclude='backend/data' \
    --exclude='./data' --exclude='*.h5' --exclude='*.zip' --exclude='*.tar' \
    --exclude='pipeline_ui/screens' --exclude='pipeline_ui/charts' \
    -czf /tmp/watercast_code.tar.gz .
scp -P 10322 /tmp/watercast_code.tar.gz ssteam@semyeongsoft.com:~/ssteam/watercast/
ssh -p 10322 ssteam@semyeongsoft.com \
  'cd ~/ssteam/watercast && cp .env /tmp/env.bak && tar -xzf watercast_code.tar.gz \
   && rm watercast_code.tar.gz && cp /tmp/env.bak .env && chmod 600 .env \
   && chmod +x docker/*.sh && docker compose up -d --build'
```

`--exclude='./data' --exclude='*.h5' --exclude='*.zip' --exclude='*.tar'`가 중요하다(루트에 인수 zip
23 GB·도커 tar 11 GB가 있다). UI 런타임은 `data/`를 읽지 않는다.
두 UI는 `ui_next/assets/`와 `pipeline_ui/assets/`의 파생 자산만 읽으므로 코드 tar에는
이것만 담고, 대용량 원본은 아래처럼 따로 올린다.
(같은 이유로 `.dockerignore`에도 넣어 빌드 컨텍스트가 커지지 않게 했다.)

## 설치 패키지로 배포 (2026-10-20 검증)

소스를 빌드하지 않고 **이미지 + 런타임 데이터 + 스크립트** 묶음으로 설치하는 경로. 다른 기관·현장 PC 에 주는 형태이며,
2026-10-20 에 semyeongsoft 서버에서 운영본(18000)과 나란히 설치해 스모크 테스트 18/18 을 확인했다.

```bash
# 1) 패키지 만들기 (로컬, 저장소 루트, docker 필요 — 약 10분, 433 MB)
scripts/package/build_release.sh                 # docker build → docker save → 데이터·스크립트·문서 → MANIFEST.json·SHA256SUMS
#   --no-build    기존 watercast:<ver> 이미지 재사용      --with-wbms   실탐지 묶음(wbms:0.13 11 GB + 모델·전처리) 별도 tar
ls dist/watercast-0.2.0-20261020.tar.gz

# 2) 서버로 보내기
scp -P 10322 dist/watercast-0.2.0-20261020.tar.gz ssteam@semyeongsoft.com:~/ssteam/pkg_test/

# 3) 서버에서 설치 (받는 쪽은 docker·docker compose v2·python3·curl 만 있으면 된다)
ssh -p 10322 ssteam@semyeongsoft.com
cd ~/ssteam/pkg_test && tar -xzf watercast-0.2.0-20261020.tar.gz && cd watercast-0.2.0-20261020
cp .env.example .env
#   운영본과 나란히 띄울 때는 포트·이름을 바꾼다(운영본은 18000/8501/8502, watercast)
sed -i 's/^WATERCAST_API_PORT=.*/WATERCAST_API_PORT=18100/; s/^WATERCAST_UI_PORT=.*/WATERCAST_UI_PORT=8511/; s/^WATERCAST_PIPE_PORT=.*/WATERCAST_PIPE_PORT=8512/; s/^WATERCAST_NAME=.*/WATERCAST_NAME=watercast-pkg/' .env
#   KMA_API_KEY·RUN_AS(id -u:id -g)·BACKEND_LIAISON_* 도 여기서
./install.sh            # 요구사항 → SHA256 → .env → docker load → compose up → 헬스 대기 → smoke_test.sh 18개
./status.sh             # 상태        ./smoke_test.sh  다시 검사
./stop.sh               # 중지(데이터·이미지 유지)      ./start.sh  재기동(.env 반영)
./uninstall.sh --images # 제거(이미지까지), --data 면 잡 상태·산출물도
```

- 패키지의 이미지 태그는 `watercast:0.2.0`(운영본은 `watercast:latest` 로 빌드) — 같은 서버에서 겹치지 않는다. 같은 태그가 이미 있으면 `install.sh` 는 적재를 건너뛰므로 새 패키지를 시험할 때는 `docker image rm watercast:0.2.0` 뒤에 설치한다.
- 서버의 시험 설치는 `~/ssteam/pkg_test/watercast-0.2.0-20261020`(18100/8511/8512, 컨테이너 `watercast-pkg`·`watercast-pkg-pipeline`). 필요 없으면 그 디렉터리에서 `./stop.sh` 또는 `./uninstall.sh --images`.
- 오류 응답은 모든 API 가 같은 봉투(`detail` + `error{status, code, message_ko, fields[], request_id, …}`, 헤더 `X-Request-ID`)로 답하고 `GET /api/v1/errors/recent` 에 최근 200건이 남는다. 서버 로그와는 `docker compose logs watercast | grep <request_id>` 로 맞춘다.

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

**(2026-10-03 갱신)** 이 결정은 뒤집었다 — 소켓을 마운트하고 이미지에 docker CLI 를 넣어 API 컨테이너가
직접 탐지한다. 아래 "컨테이너 안에서 탐지 실행 (docker 소켓)" 절 참고.

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
`docker` CLI가 없어서다(위 "남은 블로커" 참조). 2026-10-03 이후 compose 는 docker 소켓을 마운트하므로
`image_available` 은 실제 값이고 탐지 잡도 컨테이너 안에서 돈다(아래 절).

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

## 컨테이너 안에서 탐지 실행 (docker 소켓, 2026-10-03)

위 "남은 블로커"(API 컨테이너 안에 `docker` 가 없다)는 해소됐다. 이미지에 docker CLI(클라이언트 정적
바이너리 28.3.3, Dockerfile 1단계에서 꺼내 `/usr/local/bin/docker` 하나만 복사)를 넣고 compose 가 호스트
`/var/run/docker.sock` 을 마운트하므로, 탐지 잡(`POST /api/v1/wbms/segmentation/jobs`)이 캐시에 없거나
`force=true` 면 백엔드가 **호스트 데몬으로 `wbms:0.13` 을 직접 띄운다**. `GET /api/v1/wbms/status` 의
`image_available` 도 이제 실제 값이다. 호스트 러너 `scripts/wbms_chain_runner.py`(패키지 `detect_scene.sh`)는
그대로 대안으로 남는다.

compose(`docker-compose.yml`, 설치 패키지는 `packaging/docker-compose.yml`)의 `watercast` 서비스에 더해진 것:

```yaml
    environment:
      - BACKEND_HOST_ROOT=${HOST_ROOT:-${PWD}}     # 이 저장소의 호스트 절대경로
      - BACKEND_CONTAINER_ROOT=/app
      - BACKEND_RUN_AS=${RUN_AS:-1000:1000}        # wbms 컨테이너 --user (산출물 소유자)
      - BACKEND_WBMS_GPU=${WBMS_GPU:-auto}
    volumes:
      - ./data/wbms_runs:/app/data/wbms_runs       # 마스크 캐시만 쓰기 가능 — 중첩 바인드가 상위 :ro 를 덮는다
      - /var/run/docker.sock:/var/run/docker.sock
```

- **경로 변환 (`HOST_ROOT`)**: `docker run -v` argv 는 컨테이너 안에서 조립되지만 실행은 호스트 데몬이 하므로
  `-v /app/data/...` 는 호스트에 없는 경로다. `backend/app/wbms/runner.py` 의 `docker_prefix()` 가
  `BACKEND_CONTAINER_ROOT`(`/app`) 아래 경로를 `BACKEND_HOST_ROOT` 기준 호스트 경로로 바꿔 넘긴다(심볼릭
  링크는 먼저 풀고, `/app` 밖 경로는 그대로). 변환표는 `WbmsSettings.from_env()` 가 `runner.set_path_map()` 으로
  모듈 전역에 넣는다(어댑터는 설정 객체 없이 `docker_prefix()` 를 부르기 때문). 기본값 `${PWD}` 는 compose 를
  실행한 디렉터리라 저장소 루트에서 `docker compose up` 하면 맞고, 다른 곳에서 띄우면 `.env` 에
  `HOST_ROOT=/절대경로` 를 적는다. 비어 있으면 변환하지 않아 호스트에서 백엔드를 직접 돌리는 경우와 같다.
- **소유자 (`RUN_AS`)**: API 컨테이너 안의 프로세스는 root 라 그대로 두면 wbms 산출물이 호스트에서 root
  소유가 된다. `BACKEND_RUN_AS=uid:gid` 가 wbms 컨테이너의 `--user` 를 호스트 사용자로 고정한다.
  `data/wbms_runs` 소유자와 같아야 쓸 수 있다(서버 ssteam 계정이면 `id -u`:`id -g`).
- **GPU auto**: API 컨테이너에는 `nvidia-smi` 가 없어 프리플라이트를 `docker run --rm --gpus device=0
  --entrypoint nvidia-smi wbms:0.13 --query-gpu=memory.free ...`(호스트 데몬, 60초 제한)로 한다. 여유 VRAM
  ≥ 3000 MiB 면 `--gpus device=0` 으로 실행(30~60초), 아니면 잡 `warnings` 에 사유를 남기고 CPU(10분 내외).
  nvidia-container-toolkit 이나 소켓이 없으면 프리플라이트가 None → CPU. `WBMS_GPU=gpu` 는 부족 시 잡 거절.
- **보안**: docker 소켓은 컨테이너에 호스트 docker 권한(사실상 호스트 root)을 준다. 이 과제는 단일 서버·
  내부망·인증 없는 API 라는 전제라 받아들였다(`security-findings` 메모 참고). API 가 외부에 노출되는 구성이면
  compose 의 소켓 줄을 빼고 호스트 러너로 돌아간다 — 그러면 잡은 캐시만 답하고 `image_available` 은 false 다.
- 검증: `backend/tests/test_wbms.py` 의 경로 변환·`RUN_AS`·docker 경유 프리플라이트 케이스 + 실측(2026-10-03,
  로컬 RTX A5000, 이미지 `watercast:dind-test`, 저장소 루트에서 compose 기동). `force=true` 로 ICEYE 20200330 잡 →
  `succeeded`, `result.source: run`, 558초(vucatcher 상주로 여유 VRAM 2 MiB → 잡 `warnings` 에 CPU 폴백 사유, GPU 면
  30~60초), 호스트 데몬에서 본 wbms 컨테이너 `user=1000:1000` + 바인드 4개 전부 호스트 경로, 산출
  `data/wbms_runs/wb/2020/20200330/WB_Busan_ICEYE_20200330T190048.tif` uid 1000 소유, 규약 사본
  `backend/data/wbms_products/PT000003_ICEYE_X5_SLC_SM_25599_20200330T190048.tif`(API 컨테이너가 root 라 root 소유 —
  종전과 같음). CPU 재탐지 마스크는 2026-08-20 GPU 산출과 425,013,728 픽셀 중 155 픽셀만 다르다.
- 주의(어댑터 쪽, 미수정): API 의 `force=true` 는 백엔드 캐시만 건너뛰고 모듈 인자 `--force` 를 넘기지 않는다. 같은
  장면 산출물이 이미 `/out` 에 있으면 `detect_water.py` 가 "조건 불일치 — SKIP 유지" 로 0초에 exit 0 하고 백엔드는
  "no COMPLETE WB_*_qc.json" 으로 `failed` 처리한다. 재탐지하려면 그 날짜 디렉터리를 비우거나 어댑터가 `--force` 를
  전달하게 고쳐야 한다(`scripts/wbms_chain_runner.py` 는 이미 전달한다).

## 환경변수 (백엔드·UI가 읽는 것 전부)

| 변수 | 읽는 곳 | 기본값 | 용도 |
|---|---|---|---|
| `KMA_API_KEY` | weather_service, backend | 없음 → sample 모드 | 기상청 ASOS 일반 인증키 |
| `BACKEND_RESULT_DIR` | backend | `backend/data/results` | 예측 결과 저장소 |
| `PREDICTOR_PLUGINS` | backend | `""` | 마스크 시계열 예측 어댑터 플러그인 `pkg.mod:Class,…` |
| `GAUGE_FORECASTER_PLUGINS` | backend | `""` | 수위 시계열 예측기 플러그인 |
| `SEGMENTER_PLUGINS` | backend | `""` | 수체 탐지 어댑터 플러그인 |
| `REGIONS_DIR` | backend | `<repo>/regions` | 내장 지역 레지스트리 |
| `REGIONS_USER_DIR` | backend | `backend/data/regions` (compose: `/app/backend/data/regions`) | 운영 중 추가·재학습한 지역 |
| `BACKEND_WBMS_CATALOG_JSON` | backend | `backend/app/wbms/catalog.json` | 센서·테스트베드 카탈로그 |
| `BACKEND_WBMS_IMAGE` / `_DOCKER_BIN` | backend | `wbms:0.13` / `docker` | 인도 컨테이너 |
| `BACKEND_WBMS_HANDOVER_DIR` | backend | `data/incoming/handover` | 인수 패키지 |
| `BACKEND_WBMS_JOBS_DIR` | backend | `data/wbms_jobs` (compose: `/app/backend/data/wbms_jobs`) | 탐지 잡 상태 — **쓰기 가능해야 함** |
| `BACKEND_WBMS_WB_DIR` / `_FUSION_DIR` | backend | `data/wbms_runs/wb` / `…/fusion` (compose 융합: `/app/backend/data/wbms_fusion`) | 마스크 캐시(읽기) / 융합 작업공간(쓰기) |
| `BACKEND_WBMS_BUNDLE_DIR` / `_AWS_CSV` | backend | 인수 패키지 내부 | 융합 LSTM 번들·기상 CSV |
| `BACKEND_WBMS_SEGMENTATION_TIMEOUT_S` / `_FUSION_TIMEOUT_S` | backend | 3600 / 600 | 컨테이너 실행 제한 |
| `BACKEND_WBMS_GPU` / `_GPU_MIN_FREE_MIB` | backend | `cpu` / 3000 (compose: `${WBMS_GPU:-auto}`) | 추론 장치·VRAM 프리플라이트. 컨테이너 안에서는 docker 경유 nvidia-smi |
| `BACKEND_HOST_ROOT` / `BACKEND_CONTAINER_ROOT` | backend | 없음 → 변환 안 함 / `/app` (compose: `${HOST_ROOT:-${PWD}}` / `/app`) | docker 소켓으로 호스트 데몬에 넘기는 `-v` 경로 변환 — "컨테이너 안에서 탐지 실행" 절 |
| `BACKEND_RUN_AS` | backend | 없음 → 프로세스 uid:gid (compose: `${RUN_AS:-1000:1000}`) | wbms 컨테이너 `--user` — 산출물 소유자 |
| `BACKEND_WBMS_PRODUCT_DIR` | backend | `data/wbms_runs/products` (compose: `/app/backend/data/wbms_products`) | 규약 파일명 `{product_type}_{product_id}.tif` 복사본 — **쓰기 가능해야 함** |
| `BACKEND_LIAISON_LOG_URL` | backend, chain runner | 없음 → 전송 안 함 | 통합 플랫폼 알고리즘 로그 API 주소 |
| `BACKEND_LIAISON_SYSTEM_CODE` / `_ALGM_ID` | backend, chain runner | `SC000000` / `AG000000` (자리표시) | 로그의 system_code·algm_id — 통합 측 배정값으로 교체 |
| `BACKEND_LIAISON_TIMEOUT_S` | backend | 5 | 로그 전송 제한 시간 |
| `SMSS_NET` | docker-compose.smss-net.yml | `smss-net` | 3세부와 공유하는 외부 도커 네트워크 이름 |
| `WATER_API_URL` | ui_next, entrypoint | `http://localhost:8000` (컨테이너 `http://127.0.0.1:8000`) | UI가 부르는 백엔드 |
| `BACKEND_HOST`/`BACKEND_PORT`/`UI_HOST`/`UI_PORT` | entrypoint, run.sh | `0.0.0.0`/8000/`0.0.0.0`/8501 | 바인드 주소 |

컨테이너 안에서 `/app/data`는 읽기전용이다. 쓰기가 필요한 것(잡 상태·융합 작업공간·
사용자 지역·학습 산출물)은 전부 `backend/data`(호스트 바인드) 아래로 보낸다 — docker-compose.yml에
이미 설정되어 있다.

## 지역 레지스트리와 수위 예측 (2026-09-08)

지역(`regions/<id>/region.json`)은 위성 장면과 수위계 시계열 중 있는 것만 넣으면 해당
기능이 열린다. 내장 지역 2개(`nakdong_busan` 위성+수위계, `han_river` 수위계만)와 학습
산출물(`regions/*/models/gauge/`)이 저장소에 있어 이미지에 그대로 들어간다. 운영 중 API로
지점을 올리거나 재학습하면 `REGIONS_USER_DIR`에 저장되고 그쪽이 우선한다.

```bash
# 저장소에서 재학습(산출물을 regions/ 에 씀, 지점당 5~15초)
python3 scripts/train_gauge_models.py
# 서버 API로 특정 지점만 후보를 바꿔 재학습
curl -X POST http://localhost:18000/api/v1/regions/nakdong_busan/gauge/train \
  -H 'Content-Type: application/json' -d '{"stations":["구포대교"],"forecasters":["analog_knn"]}'
```

## 과제 통합 플랫폼 연계 (2026-09-29)

다른 세부와 주고받기 위해 붙인 것 네 가지. 시트·근거는 `docs/liaison/연계정리_이노뎁_2세부_작성본.md`.

**1. 연계 파라미터 6종** — 탐지 잡 `POST /api/v1/wbms/segmentation/jobs` 본문과 장면 등록
`POST /api/v1/regions/{id}/satellite/scenes` Form, 배치 러너 `scripts/wbms_chain_runner.py` 인자로
`order_id · scene_id · work_id · algm_id · product_type · product_id` 를 받는다. 전부 선택이며 잡 문서에 그대로 남는다.

**2. 규약 파일명 산출물** — 탐지가 끝나면 마스크 GeoTIFF 를 `{product_type}_{product_id}.tif` 이름으로
`BACKEND_WBMS_PRODUCT_DIR` 에 하나 더 둔다(하드링크·복사, 원본 `WB_*.tif` 는 유지). `product_id` 를 안 주면
전처리 사이드카 `_meta.json` 의 `Provenance.input_file` 줄기(예 `ICEYE_X5_SLC_SM_23467_20200302T183857`)에서
파생한다. 잡 결과의 `product_tif` 로 경로를 알 수 있고 `GET …/jobs/{job_id}/mask` 로 내려받는다.

**3. 알고리즘 로그** — `BACKEND_LIAISON_LOG_URL` 이 있으면 잡 상태가 바뀔 때마다 규격 JSON 을 POST 한다.
대기 WS000000 → 진행 WS000001 → 완료 WS000002, 실패는 WS000003(LL000004). 캐시 히트는 완료 1건만.
전송 실패는 잡 `warnings` 에만 남고 탐지 결과에는 영향이 없다. system_code·algm_id 는 통합 측 배정 전이라
자리표시 `SC000000`/`AG000000` 이다 — 배정되면 `.env` 에 넣고 `docker compose up -d`.

**4. smss-net 직결** — 3세부와 같은 도커 네트워크가 생기면
`docker compose -f docker-compose.yml -f docker-compose.smss-net.yml up -d` 로 올린다. 그러면 3세부
컨테이너에서 `http://watercast:8000/api/v1/…` 로 공개 도메인 없이 부를 수 있다. 네트워크는 먼저 있어야 한다
(`docker network create smss-net`, 생성 주체는 협의).

**3세부가 받아 가는 API 보강** — 마스크 GeoTIFF 다운로드 `GET /api/v1/wbms/segmentation/jobs/{job_id}/mask`,
`…/jobs/{job_id}/files/{name}`, `GET /api/v1/regions/{id}/satellite/scenes/{scene_id}/mask`;
`GET …/satellite/levels` 와 `…/satellite/area-series` 에 `start`·`end` 기간 필터.

## 설치 패키지 만들기 (다른 기관에 전달, 2026-10-01)

소스 빌드 없이 `docker load` 로 올리는 묶음. 템플릿은 `packaging/`, 빌더는 `scripts/package/build_release.sh`.

```bash
scripts/package/build_release.sh               # dist/watercast-<ver>-<날짜>.tar.gz  (약 430 MB)
scripts/package/build_release.sh --with-wbms   # + dist/…-detect.tar.gz  실탐지 묶음 (wbms:0.13 + 모델·전처리·보조자료, 약 23 GB)
scripts/package/build_release.sh --no-build    # 이미지는 그대로, 스크립트·데이터만 다시 묶기
```

받는 쪽은 `tar -xzf` 뒤 `./install.sh` 한 번이다(요구사항 확인 → 체크섬 → .env → load → up → 헬스 → 스모크 10개).
실탐지는 `-detect` 묶음을 `--strip-components=1` 로 겹쳐 푼 뒤 `./install.sh --with-wbms`, `./detect_scene.sh <YYYYMMDD> --force`.
자세한 것은 `packaging/README_설치.md`. 묶음 내용·검증 기록은 작업로그 6-13절.

주의: 패키지 compose 는 `packaging/docker-compose.yml` 로 따로 있다. 개발 compose 에 환경변수를 더하면
`backend/tests/test_packaging.py` 가 패키지 compose 에도 있는지 검사한다. `.dockerignore` 가 DEPLOY.md·docs·packaging 을
이미지에서 뺀다 — 코드가 읽는 파일을 그쪽에 두지 말 것.

## 입구 하나짜리 장면 파이프라인과 쉬운 화면 (2026-10-03)

`POST /api/v1/scenes`(multipart) 하나로 장면 등록 → 수체 탐지 → 면적 → 수위(③ calc_wlwa) → 시계열 예측 →
규약 파일·로그 를 백그라운드로 수행한다(`backend/app/pipeline/`). 상태는 `GET /api/v1/scenes/{scene_id}` 의
6단계(pending·running·done·skipped·failed), 산출은 `backend/data/scenes/<region>/<scene_id>/` 한 폴더.
기존 단계별 API 는 그대로다. 쉬운 화면은 `/easy`(`backend/app/static/easy/index.html`, 외부 의존 없음).

- 새 영상을 올린 것이므로 같은 날짜 완료 마커가 있어도 **항상 재탐지**한다(`force` → 모듈 `--force`).
- ICEYE 는 날짜별 3-class 마스크가 없으면 가장 가까운 날짜 것을 대신 쓰고 경고를 남긴다.
- 수위 단계는 날짜별 DEM(`06_aux/DEM_busan_ngii3m/icegrid/<date>_DEM_32652.tif`)이 있어야 하고, 없으면 skipped.
- wbms 컨테이너가 쓰는 작업 디렉터리는 0o777 로 만든다 — API 컨테이너는 root, wbms 는 `RUN_AS` 로 돌기 때문.
- 서버 실측 2026-10-03(4090): 등록 4 s · 탐지 29 s(GPU) · 면적 3 s · 수위 2 s · 예측 0.03 s — 합계 42 s.
  로컬 CPU 폴백은 탐지 481 s.

## 화면 구성 (2026-10-14 개편)

| 주소 | 화면 | 대상 |
|---|---|---|
| `/` → `/app` | **서비스** — 자료 넣기·위치 확인 → 탐지(수체·수위) → 시각화 → 예측(모델 선택) | 운영자 |
| `/eval` | **모델 평가** — 지역 → 수위 자료(WAMIS) 수집 → 기상(ASOS) 수집 → 학습 → 성능 평가 → 시험 | 데이터 담당 |
| `/easy` | 쉬운 화면(단일 업로드) | 비전문가 |
| `/docs` · :8501 · :8502 | API 문서 · 전문가 Streamlit · 파이프라인 데모 | 도움말 탭에서 링크 |

정적 단일 파일(`backend/app/static/{app,eval,easy}/index.html`), 외부 의존 없음. 새 API: `/regions/{id}/geometry`, `/regions/{id}/readiness`,
`/wamis/stations`, `/regions/{id}/gauge/collect`(+jobs), `/regions/{id}/weather/collect`(+jobs), `POST /scenes/{id}/predict`
(`backend/app/screens_api.py`, `backend/app/screens/`). 캡처 스크립트 `scripts/capture_{service,eval,easy}_ui.py`(스텁), `capture_*_real.py`(실서버).

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
# (httpx2 는 starlette 최신 TestClient 가 요구하는 패키지다 — 오타가 아니다)
```

`pytest`와 `httpx2`는 테스트 전용이라 이미지에 넣지 않았다. 그래서 운영 컨테이너에서
`unittest discover`를 돌리면 `ModuleNotFoundError: pytest`로 8건이 실패하는데, 이는
정상이며 위 일회용 컨테이너 방식으로 확인한다.
