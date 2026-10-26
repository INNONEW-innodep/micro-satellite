# WATERCAST 설치 안내 — 2세부 이노뎁 · 수체·수위 탐지 및 시계열 예측

이 묶음은 2세부 서비스 전체 — **플랫폼 런타임 이미지 + 앱 바이너리(코드 묶음) + 수체 탐지 엔진(wbms 이미지·모델·보조자료) + 데이터 + 스크립트**다. 소스를 빌드하지 않는다. 1세부가 새 위성 영상을 보내면 탐지 → 면적 → 수위 → 예측까지 이 묶음 하나로 돈다.
설치는 `./install.sh` 한 번이고, 끝나면 스모크 테스트 18개가 자동으로 돈다. 배포 담당자용 명령 순서는 `배포_실행순서.md`.

- `images/watercast-runtime-<버전>.tar.gz` — 플랫폼 런타임 이미지(python 3.11 + 의존성 + docker CLI). **앱 코드가 없다.** 의존성이 바뀔 때만 교체.
- `images/wbms-0.13.tar.gz` — 수체 탐지 엔진 이미지(11 GB) + `data/incoming/handover/{03_model,05_l1_pre,06_aux}`(모델 가중치·전처리 입력 영상·보조자료). `install.sh` 가 자동 적재한다. 탐지 재빌드본을 받으면 이미지만 교체.
- `app/watercast-app-<버전>-<빌드>.tar.gz` — 앱 바이너리(backend·화면·지역 자료·스크립트 + `VERSION.json`). `install.sh` 가 `app/current` 에 풀어
  컨테이너 `/app` 에 마운트한다. **코드가 바뀌면 이 묶음만 받아 `./update_app.sh` 로 교체**한다(이미지 그대로, 1분).

## 1. 요구사항

| 항목 | 필요 |
|---|---|
| OS | Linux x86_64. Rocky Linux 9 · Ubuntu 22.04 이상에서 확인 |
| docker | 24 이상, `docker compose` v2 플러그인 |
| python3 | 3.9 이상. 표준 라이브러리만 쓴다. 스모크 테스트·실탐지 러너용 |
| tar, curl, sha256sum | 묶음 풀기·설치 스크립트가 쓴다 |
| 디스크 | 전체 패키지: tar 약 40 GB, 풀고 이미지(런타임 1 GB + 탐지 11 GB)까지 적재하면 약 90 GB. lite 묶음(탐지 엔진 제외)은 2 GB(wbms 이미지 11 GB + 모델·전처리·보조자료 32 GB. 받은 tar 를 지우기 전까지는 그만큼 더) |
| 포트 | 18000 API · 8501 업무 UI · 8502 데모 UI. 겹치면 `.env` 에서 바꾼다 |
| GPU | 필요 없음. 실탐지를 GPU 로 돌릴 때만 NVIDIA 드라이버 + nvidia-container-toolkit |

docker 그룹에 들어 있지 않으면 스크립트를 `sudo` 로 실행한다. 이후 `start/stop/uninstall/detect_scene` 도 같은 계정으로 돌린다
(`.env` 와 `backend/data` 가 root 소유가 된다 — `.env` 편집도 `sudo`). 환경변수를 줄 때는 `sudo WITH_SMSS_NET=1 ./start.sh` 처럼 `sudo` 뒤에 쓴다.

## 2. 설치

```bash
tar -xzf watercast-<버전>-<날짜>.tar.gz
cd watercast-<버전>-<날짜>
./install.sh
```

`install.sh` 가 하는 일: 요구사항 확인 → 체크섬 확인 → `.env` 생성 → 런타임 이미지·탐지 엔진 이미지 `docker load` → 앱 묶음을 `app/current` 에 적재
→ `docker compose up -d` → API 헬스 대기(최대 3분) → `smoke_test.sh`. 출력은 `logs/install_<시각>.log` 에도 남고 실패하면 멈춘 단계·명령과 볼 로그를 찍는다. 다시 실행해도 안전하다 (있는 `.env`·데이터는 그대로, 이미지는 다시 적재하지 않는다).

끝나면 이렇게 뜬다.

```
설치 완료
  API 문서   http://<이 서버>:18000/docs
  업무 UI    http://<이 서버>:8501/
  데모 UI    http://<이 서버>:8502/
```

## 3. 설정 바꾸기 — `.env`

| 변수 | 기본 | 뜻 |
|---|---|---|
| `WATERCAST_API_PORT` · `WATERCAST_UI_PORT` · `WATERCAST_PIPE_PORT` | 18000 · 8501 · 8502 | 호스트 포트 |
| `COMPOSE_PROFILES` | `demo` | 데모 UI 까지 띄움. 비우면 API·업무 UI 만 |
| `KMA_API_KEY` | 비움 | **선택.** 기상청 공공데이터포털 일반 인증키 — 과거 관측(ASOS)·단기예보를 새로 받을 때만. 없어도 예측·탐지·화면 전부 동작. 우리 API 자체는 인증이 없다 |
| `WATERCAST_NAME` | watercast | 컨테이너 이름 접두. 같은 서버에 두 벌 띄울 때 바꾼다 |
| `BACKEND_LIAISON_LOG_URL` | 비움 | 통합 플랫폼 알고리즘 로그 API. 비면 전송 안 함 |
| `BACKEND_LIAISON_SYSTEM_CODE` · `BACKEND_LIAISON_ALGM_ID` | SC000000 · AG000000 | 수체탐지 코드. 통합 측 배정값으로 교체 |
| `WITH_SMSS_NET` | 0 | 1 이면 3세부 공유 네트워크 오버레이(`docker-compose.smss-net.yml`)를 항상 덧붙인다 |
| `HOST_ROOT` | 비움 → `./start.sh` 를 실행한 디렉터리 | 이 패키지 디렉터리의 호스트 절대경로. 탐지 컨테이너 마운트 경로 변환용(6절) |
| `RUN_AS` | 1000:1000 | 탐지 컨테이너 uid:gid = 산출물 소유자. `data/wbms_runs` 소유자(`id -u`:`id -g`)와 맞춘다 |
| `WBMS_GPU` | auto | 탐지 장치. auto 는 여유 VRAM 3000 MiB 이상이면 GPU, 아니면 CPU 폴백. cpu / gpu(부족하면 잡 거절) |
| `SMSS_NET` | smss-net | 그 네트워크 이름 |

바꾼 뒤 `./start.sh` 로 재기동한다. 값은 따옴표가 있어도 없어도 되고, 윈도 편집기(CRLF)로 고쳐도 된다.
`COMPOSE_PROFILES` 를 비우고 `./start.sh` 하면 데모 컨테이너는 내려간다.

## 4. 운영 명령

| 명령 | 뜻 |
|---|---|
| `./status.sh` | 컨테이너 상태와 API 헬스 |
| `./start.sh` | 기동·재기동 (`.env` 반영) |
| `./stop.sh` | 중지(데모 컨테이너 포함). 이미지·데이터는 남는다 |
| `./smoke_test.sh` | 설치 확인 18개 다시 실행 |
| `./gpu_check.sh [--quick]` | **GPU 점검** — 호스트 드라이버 → nvidia-container-toolkit → 탐지 이미지 안 GPU → TensorFlow 인식 → 합성곱 실험 순으로 보고 `WBMS_GPU` 권장값을 낸다(새 세대 GPU 대비). 로그 `logs/gpu_check_*.log` |
| `./postcheck.sh [--quick]` | **배포 후 확인** — 환경·설정·컨테이너·API·스모크를 점검하고 `logs/postcheck_<시각>.txt` 보고서. 실패 시 컨테이너 로그를 붙인다 |
| `./update_app.sh app/<앱 묶음>.tar.gz` | **앱 바이너리 교체** — 이미지는 그대로, 코드만 교체 → 재기동 → 헬스 → 스모크. `--rollback` 으로 직전 build 복귀 |
| `./uninstall.sh [--images] [--wbms-image] [--data]` | 제거. `--wbms-image` 는 탐지 엔진 11 GB 이미지(다른 작업과 공유될 수 있어 따로 지정). `--data` 는 잡 상태·규약 산출물·사용자 지역을 지운다 |
| `WITH_SMSS_NET=1 ./start.sh` | 3세부 공유 네트워크에 붙여 기동(1회). 계속 붙여 두려면 `.env` 에 `WITH_SMSS_NET=1` 을 적는다 — 환경변수로만 주면 다음 `./start.sh` 때 오버레이 없이 재생성된다. 네트워크(`docker network create smss-net`)가 먼저 있어야 한다 |

## 4-0. 화면 세 가지

| 주소 | 무엇 |
|---|---|
| `http://<서버>:18000/` (→ `/app`) | **서비스 화면** — 영상 넣고 위치 확인 → 물 영역·수위 → 그림·그래프 → 원하는 모델로 예측 |
| `http://<서버>:18000/eval` | **모델 평가 화면** — 새 지역 만들고 수위·기상 자료를 모아 학습하고 성능을 본다 |
| `http://<서버>:18000/easy` | 쉬운 화면 — 영상 한 장 올려 결과만 받는다 |

## 4-1. 가장 쉬운 사용법 — 쉬운 화면

브라우저에서 `http://<서버>:18000/easy` 를 연다. 세 단계다.

1. 지역·영상 종류·날짜를 고른다.
2. 위성 영상 파일(.tif)을 끌어다 놓고 **시작하기**.
3. 진행 상황이 체크리스트로 올라가고, 끝나면 물 영역 그림·면적·내려받기 버튼이 나온다.

서버가 영상 등록 → 물 영역 찾기 → 면적 계산 → 수위 계산 → 앞으로의 변화 예측 → 결과 파일 정리를
순서대로 다 한다. GPU 가 있으면 1분 안쪽, 없으면 10분쯤 걸린다. 같은 일을 API 로 하려면
`POST /api/v1/scenes` 에 같은 항목을 보내고 `GET /api/v1/scenes/{scene_id}` 로 진행을 본다.
결과는 장면마다 `backend/data/scenes/<지역>/<장면>/` 폴더 하나에 모인다(규약 파일 `PT000003_*.tif`,
물 영역 원본, 미리보기, 면적·수위·예측 JSON, 상태 기록).

## 5. 연계에 쓰는 API

주소는 `http://<서버>:18000`. 전체 목록은 `/docs`. 인증은 없다.

| 용도 | 메서드 · 경로 |
|---|---|
| 상태 | `GET /api/v1/health` |
| 장면 파이프라인 실행 | `POST /api/v1/scenes` 폼: region_id · sensor · image_date · pre_file 또는 pre_path · 연계 번호 6종 |
| 장면 진행·결과 | `GET /api/v1/scenes/{scene_id}` · `GET /api/v1/scenes` · `…/files/{name}` |
| 위성 장면 목록 | `GET /api/v1/regions/{region_id}/satellite/scenes` |
| 장면 마스크 GeoTIFF | `GET /api/v1/regions/{region_id}/satellite/scenes/{scene_id}/mask` |
| 위성 수위 · 실측 대조 | `GET /api/v1/regions/{region_id}/satellite/levels?station_id=&start=&end=` |
| 공통 격자 면적 시계열 | `GET /api/v1/regions/{region_id}/satellite/area-series?start=&end=` |
| 수위 전망 D+1~30 | `GET /api/v1/regions/{region_id}/gauge/{station_id}/forecast?as_of=&horizon=` |
| 수체 시계열 예측 | `POST /api/v1/predictions` → `GET /api/v1/predictions/{id}/files` |
| 탐지 잡 접수 | `POST /api/v1/wbms/segmentation/jobs` 본문에 `order_id scene_id work_id algm_id product_type product_id` 선택 |
| 잡 상태 · 규약 산출물 경로 | `GET /api/v1/wbms/segmentation/jobs/{job_id}` → `result.product_tif` |
| 잡 마스크 다운로드 | `GET /api/v1/wbms/segmentation/jobs/{job_id}/mask` · `…/files/{name}` |
| 장면 폴더의 수위 전망(연계용, 2026-10-20) | `GET /api/v1/scenes/{scene_id}/files/level_forecast.json` — 촬영일 원점, 연결 지점별 D+1~30 |
| 시 단위 수위 전망(실시간형, 2026-10-20) | `POST …/regions/{id}/gauge/collect-hourly`(WAMIS 시 자료 수집) → `GET …/gauge/{station}/forecast-hourly?as_of=&horizon_hours=1..48` |
| 기상청 단기예보·댐 운영 자료(2026-10-20) | `POST …/regions/{id}/weather/forecast-collect`(KMA_API_KEY 에 단기예보 서비스 신청 필요) · `POST …/regions/{id}/dam/collect` · `GET …/dam` |
| 최근 오류 조회(운영 점검, 2026-10-20) | `GET /api/v1/errors/recent?limit=50&min_status=400` |

`region_id` 는 `nakdong_busan`, `geum_daecheong` 등. `scene_id` 는 `20200302_ICEYE` 꼴.
탐지 잡은 이미 탐지된 장면이면 즉시 `succeeded` 로 답하고, 규약 파일명
`PT000003_{product_id}.tif` 사본을 `backend/data/wbms_products/` 에 만든다. 알고리즘 로그는
`.env` 의 `BACKEND_LIAISON_LOG_URL` 이 있을 때 대기·진행·완료·에러 4종을 JSON 으로 POST 한다.

## 6. 새 영상 탐지

탐지 엔진(wbms:0.13 이미지·모델·보조자료)은 이 패키지에 들어 있고 `install.sh` 가 적재한다. API 컨테이너가 호스트 docker 소켓
(`/var/run/docker.sock`, compose 에 들어 있다)으로 wbms 컨테이너를 직접 띄우므로 **새 영상을 올리면 API 가 스스로 탐지한다.**

```bash
./gpu_check.sh                                                   # GPU 로 돌릴 수 있는지 판정(없거나 실패하면 CPU, 장면당 10분 내외)
curl -s -X POST http://localhost:18000/api/v1/scenes -F region_id=nakdong_busan -F sensor=ICEYE -F image_date=2020-04-16 \
  -F acquisition_time=183630 -F pre_path=/app/data/incoming/handover/05_l1_pre/iceye_pre/2020/20200416/Pre_Busan_ICEYE_20200416T183630.tif \
  -F order_id=21 -F scene_id=5 -F work_id=W-21                     # 장면 파이프라인(탐지→면적→수위→예측→납품) 접수
curl -s http://localhost:18000/api/v1/scenes/20200416_ICEYE        # 단계별 진행·경고 (detect 단계 시간으로 GPU/CPU 구분)
```

새 영상은 전처리 결과 `data/incoming/handover/05_l1_pre/<센서>_pre/<yyyy>/<yyyymmdd>/Pre_Busan_<센서>_<yyyymmdd>T<시각>.tif` 에 두고
`pre_path=/app/data/incoming/...` 로 부르거나, `pre_file` 로 업로드한다(그러면 경로가 필요 없다). `/simple` 화면의 "새 영상 올리기"도 같은 일을 한다.

`.env` 의 세 값이 탐지 실행을 조정한다 — `HOST_ROOT`(패키지 디렉터리 절대경로. 비우면 `./start.sh` 를 실행한 디렉터리라 보통 그대로),
`RUN_AS`(산출물 소유자 uid:gid. `data/wbms_runs` 소유자와 같아야 쓸 수 있다), `WBMS_GPU`(auto: GPU 가 되면 GPU 30~60초, 안 되면 CPU 10분 내외.
GPU 실행이 실패해도 CPU 로 자동 재시도). 잡의 `warnings`·`gpu_fallback` 에 CPU 사유가 남는다.
소켓은 컨테이너에 호스트 docker 권한을 주므로 내부망 서버 전제다 — 빼려면 compose 의 `/var/run/docker.sock` 줄을 지운다(그러면 탐지 잡은 캐시된 장면만 답한다).

호스트에서 직접 돌리는 대안(소켓을 뺐거나 API 를 거치지 않을 때):

```bash
./detect_scene.sh 20200302 --force                       # CPU 10분 내외. 묶음에 든 장면은 이미 캐시라 --force 없이는 건너뛴다
./detect_scene.sh 20200302 --force --gpu auto --order-id 1 --scene-id 1 --work-id 1
```

산출: `data/wbms_runs/wb/<yyyy>/<date>/WB_*.tif` 와 `backend/data/wbms_products/PT000003_*.tif`.

## 6-1. 오류 응답 읽는 법 (2026-10-20)

잘못된 요청은 상태코드와 함께 같은 모양의 JSON 으로 온다. `detail` 은 예전 그대로이고 `error` 가 붙는다.

```json
{"detail": [...], "error": {"status": 422, "code": "validation_error",
  "message_ko": "입력 2곳이 잘못됐습니다: body.dam_code, body.start",
  "fields": [{"loc": "body.dam_code", "msg_ko": "형식이 맞지 않습니다 (패턴 ^\\d{7}$)", "input": "12", "hint_ko": "요청 본문(JSON)의 해당 항목을 고쳐 다시 보내세요"}],
  "request_id": "a1b2c3d4e5f6", "method": "POST", "path": "/api/v1/regions/x/dam/collect", "timestamp": "2026-10-20T12:00:00Z",
  "docs_url": "/docs", "recent_errors_url": "/api/v1/errors/recent"}}
```

- `message_ko` 한 줄만 보면 무엇이 잘못됐는지 알 수 있고, `fields[]` 가 항목별 사유·받은 값·고치는 법이다.
- 응답 헤더 `X-Request-ID` = `error.request_id`. 서버 로그(`docker compose logs watercast | grep <id>`)와 `GET /api/v1/errors/recent` 에서 같은 id 로 찾는다.
- 500 은 `error.exception` 에 예외 이름만, traceback 은 서버 로그와 `errors/recent` 에만 남는다.
- 배경 잡(탐지·수집·학습·장면 파이프라인)은 상태 조회 응답의 `error`·`warnings`·`steps[].detail` 에 사유가 남는다(한국어).

## 7. 문제가 생기면

- **스크립트가 `✘ 실패: <명령> (<스크립트> N행)` 으로 멈춤**: 그 아래 로그 파일 경로와 컨테이너 로그 30줄이 찍힌다. `logs/` 의 해당 로그를 보낸다.
- **앱 교체 뒤 이상**: `./update_app.sh --rollback` 으로 직전 build 복귀(`app/prev`).
- **포트 사용 중** (`install.sh` 가 `호스트 포트 N 가 이미 사용 중` 으로 멈추거나 compose 가 `address already in use`): `.env` 포트를 바꾸고 `./start.sh`.
- **API 가 3분 안에 안 뜸**: `docker compose logs --tail 100 watercast`.
- **SELinux 로 마운트가 안 읽힘** (Rocky·RHEL, 스모크 3·4번 실패): `docker-compose.yml` 의 볼륨 세 줄 끝에 `:z` 를 붙이고 `./start.sh`
  (`./backend/data:/app/backend/data:z`, `./data:/app/data:ro,z` — `watercast` 서비스 2줄, `pipeline-ui` 서비스 1줄).
  compose 파일은 체크섬 검사에서 제외라 `./install.sh` 를 다시 돌려도 된다.
  호스트 실탐지(`detect_scene.sh`)의 `docker run` 마운트에는 `:z` 가 없으니 그때도 막히면 `chcon -Rt container_file_t data backend/data`.
- **체크섬 불일치**: 전송이 깨진 것. 다시 받는다.
- **탐지 잡이 422 `no ① Pre_*.tif found …`**: 그 장면의 전처리 입력이 `data/incoming/handover/05_l1_pre/...` 에 없다 — 6절의 경로 규칙대로 두었는지 확인.
- **탐지 잡이 `failed` 에 `docker executable not found` 또는 `Cannot connect to the Docker daemon`**: compose 의
  `/var/run/docker.sock` 줄이 빠졌거나 소켓 경로가 다르다(`docker context ls`). 임시로는 `./detect_scene.sh` 로 호스트에서 탐지한다.
- **탐지 잡이 `failed` 에 `/out` 쓰기 `Permission denied`**: `.env` 의 `RUN_AS` 가 `data/wbms_runs` 소유자와 다르다.
- **GPU 가 있는데 `warnings` 에 `auto 모드라 CPU 로 폴백`**: 여유 VRAM 3000 MiB 미만이거나 nvidia-container-toolkit 이 없다
  (`docker run --rm --gpus device=0 --entrypoint nvidia-smi wbms:0.13` 으로 확인).

## 8. 묶음 내용

`MANIFEST.json` 에 버전·이미지 ID·파일 수가 있고 `SHA256SUMS` 로 전부 검증한다.

| 경로 | 내용 |
|---|---|
| `images/watercast-runtime-<버전>.tar.gz` | 플랫폼 런타임 이미지(python + 의존성 + docker CLI). 앱 코드 없음 |
| `images/wbms-0.13.tar.gz` | 수체 탐지 엔진 이미지(시립대 인수, TF 2.14·CUDA 11.8, 11 GB). lite 묶음에는 없음 |
| `app/watercast-app-<버전>-<빌드>.tar.gz` (+`.sha256`) | 앱 바이너리 — API·업무 UI·데모 UI 코드, 지역 자료 7개 유역, `VERSION.json`. `install.sh`/`update_app.sh` 가 `app/current` 에 푼다(이전 build 는 `app/prev`) |
| `logs/` | 설치·교체·확인 로그와 보고서 |
| `배포_실행순서.md` | 배포 담당자용 명령 순서·설정 항목·확인·교체·장애 대응 |
| `현장_배포_절차.md` | 새 서버(Ubuntu 24.04·새 세대 GPU)에 처음부터 올리는 순서 — 호스트 준비, GPU 판정 분기, 첫 장면 시험, 연계 값 |
| `data/wbms_runs/` | 탐지 캐시 마스크 18씬 GeoTIFF + meta + qc (부산 ICEYE 4 · PlanetScope 4, 대청 PlanetScope 10) |
| `data/incoming/handover/` | 탐지 모델 가중치(03_model) · 전처리 입력 영상 22장(05_l1_pre) · 보조자료 DEM 등(06_aux) · 융합 LSTM 번들 · 기상 CSV |
| `backend/data/` | 쓰기 디렉터리. 잡 상태 · 규약 산출물 · 사용자 지역 · 예측 결과 |
| `src/` | 호스트 실탐지 러너와 연계 모듈 |
| `연계정리_이노뎁_2세부.md` | 기관 간 연계 시트 작성본 · API 목록 · 로그 포맷 대응 |
