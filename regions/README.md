# regions/ — 지역 레지스트리 (플랫폼의 데이터 입력 규약)

이 플랫폼은 **지역(region)** 단위로 동작한다. 어떤 지역이든 아래 두 종류의 자료 중
있는 것만 넣으면 그에 맞는 기능이 열린다.

| 자료 | 있으면 열리는 기능 | 형식 |
|---|---|---|
| 위성 장면 (SAR·광학 전처리 GeoTIFF) | 수체 탐지 → 면적·위성 수위 산출 (WBMS 체인) | `region.json`의 `satellite` 절 + `data/` 하위 장면 |
| 지상 수위계 일 시계열 | 지점별 수위 시계열 **예측 모델 학습·평가·전망** | `gauge/series.json` |

둘 다 있으면 위성 수위와 수위계 실측을 같은 지점에서 대조한다
(`gauge.satellite_location_links`).

## 디렉터리 규약

```
regions/<region_id>/
  region.json          # 지역 정의 (아래 스키마)
  gauge/series.json    # 지점별 일 수위 (watercast.gauge.series/1)
  models/gauge/        # 학습 산출물 — 지점별 <station>.json + evaluation.json
```

내장 지역은 저장소에 커밋되어 배포 이미지에 포함된다. 운영 중 API로 추가·수정한
지역은 `REGIONS_USER_DIR`(기본 `backend/data/regions/`, 호스트 바인드 마운트)에
저장되며 같은 `region_id`가 있으면 사용자 디렉터리가 우선한다.

## `region.json`

```json
{
  "schema": "watercast.region/1",
  "id": "nakdong_busan",
  "name": "낙동강 하구 (부산·김해)",
  "crs": "EPSG:32652",
  "weather": {"provider": "KMA_ASOS", "station_id": "159"},
  "satellite": {"available": true, "sensors": ["ICEYE", "PlanetScope"], "wbms_testbed": "busan",
                "scenes_file": "satellite/scenes.json", "levels_file": "satellite/levels.json",
                "locations": {
                  "gupo": {"name_ko": "구포대교 부근 본류 하도(64 m)", "gauge_station": "구포대교",
                           "obscd": "2022680", "river": "낙동강", "crs": "EPSG:32652",
                           "bbox_epsg32652": [499186.885, 3895504.01, 499486.885, 3895804.01],
                           "bbox_lonlat": [[128.991067, 35.205242], [128.994363, 35.205242],
                                           [128.994363, 35.202537], [128.991067, 35.202537]],
                           "center_lonlat": [128.992715, 35.203889],
                           "anchor_lonlat": [128.993333, 35.203611], "snap_distance_m": 63.8,
                           "note_ko": "OK", "source": "aoi_busan.geojson"}}},
  "gauge": {"available": true, "series_file": "gauge/series.json",
            "satellite_location_links": {"구포대교": "gupo"}},
  "forecast": {"horizon_max": 30, "horizons_report": [1,3,5,7,14,30],
               "split": {"train_end": "2022-12-31", "val_start": "2023-01-01",
                         "val_end": "2023-12-31", "test_start": "2024-01-01",
                         "test_end": "2024-12-31"}}
}
```

`satellite.locations` 는 **위성 수위 산출 위치(WBMS ③ AOI)** 다. `loc_id` 마다 투영좌표 상자(`bbox_epsg32652`, m)와
경위도 모서리(`bbox_lonlat`, 좌상→우상→우하→좌하)·중심(`center_lonlat`)·연결 수위계 지점(`gauge_station`,
`gauge.satellite_location_links` 를 뒤집은 것)·AOI 를 옮겨 온 지점 좌표(`anchor_lonlat`)와 이동 거리(`snap_distance_m`)·
인수 노트 원문(`note_ko`)을 둔다. 부산·대청은 인수 패키지의 AOI 정본
(`data/incoming/handover/02_package/deploy_package/config/aoi_{busan,dcd}.geojson`, EPSG:32652)에서
`scripts/import_satellite_locations.py` 로 옮겼다(멱등 — 다시 돌려도 같은 값이면 파일이 바뀌지 않는다).
API 는 `GET /api/v1/regions/{id}` 의 `satellite_locations` 와 `GET .../satellite/levels?station_id=` 의 `location` 으로 준다.

## `gauge/series.json`

```json
{
  "schema": "watercast.gauge.series/1",
  "unit": "m", "frequency": "daily",
  "stations": [{"id": "구포대교", "name": "구포대교", "source": "WAMIS",
                "source_code": "2022680", "terrain": "estuary",
                "lat": 35.203611, "lon": 128.993333, "address": "부산광역시 북구 구포동 1065-1 (구포대교)",
                "river": "낙동강", "tm_xy": [381554.0, 191505.0]}],
  "series": {"구포대교": [{"date": "2015-01-01", "water_level_m": 2.2758}, ...]}
}
```

- 날짜는 ISO `YYYY-MM-DD`, 일 단위. 결측일은 행을 빼면 된다(학습 시 특징 구성에서만 보간).
- `lat`·`lon`(WGS84 십진도)·`address`·`river`·`tm_xy` 는 관측소 제원이며 없어도 된다(API 는 null 로 준다).
  WAMIS 지점(`source_code`)은 `scripts/fetch_wamis_station_info.py` 가 `wl_obsinfo` 로 채운다 — 도분초(`35-12-13`)를
  십진도로 바꾸고, 한국 범위(경도 124~132·위도 33~39) 밖이거나 지역 `bbox_wgs84` 에서 50 km 넘게 떨어진 좌표는
  null 로 두고 `[이상치]` 로 보고한다. 이미 좌표가 있는 지점은 `--refresh` 없이는 다시 묻지 않는다(멱등).
- 지점 추가는 `POST /api/v1/regions/{region_id}/gauge/{station_id}/series`(CSV 또는 JSON)로도 된다.
- 예측 모델은 지점×예측일수마다 **검증 연도에서 고른 뒤 시험 연도에서 보고**한다.
  `persistence`(직전값 유지)는 skill 기준선으로만 계산하고 선택 대상에서 제외한다.

## `satellite/scenes.json`

`scripts/make_region_scene_assets.py`(일괄) 또는 `POST /api/v1/regions/{id}/satellite/scenes`(장면 하나)가 만든다.
항목마다 퀵룩(`images.input`·`overlay`·`mask`·`gt`, 긴 변 1000 px)·공통 격자 마스크(`frame`)·면적과 함께
**퀵룩 지리참조** `quicklook` 을 둔다:

```json
"quicklook": {"crs": "EPSG:32652", "source_crs": "EPSG:32652",
              "bounds_epsg32652": [473010.364, 3847364.664, 531723.364, 3921935.664],
              "width": 787, "height": 1000,
              "footprint_lonlat": [[128.702627, 35.440504], [129.349529, 35.440364], [129.34667, 34.767949], [128.705059, 34.768085]]}
```

`bounds_epsg32652` 는 원본 전처리 GeoTIFF 의 범위(= 퀵룩 PNG 네 장 모두의 범위), `width`·`height` 는 퀵룩 화소 크기다.
화면은 이 상자→화소의 아핀 매핑(`ui_next/geo_overlay.py` `to_pixel`)으로 수위계 지점·AOI 를 사진 위에 찍는다.
2026-09-09 이전에 만든 장면(부산 8·대청 10)은 `scripts/backfill_scene_georef.py` 가 `source_pre` 원본을 열어 채웠다
(원본이 파일명뿐이면 `05_l1_pre/{iceye_pre,planet_pre,dcd_pre}` 에서 찾고, 없으면 지어내지 않고 건너뛰어 보고한다).

## 새 지역 추가 절차

1. `POST /api/v1/regions` 로 지역 정의를 만들거나 위 디렉터리를 직접 만든다.
2. 수위계가 있으면 `series.json`을 넣거나 API로 지점별 시계열을 올린다.
   WAMIS(무인증) 지점이라면 `scripts/fetch_wamis_levels.py` 의 범용 경로로 바로 만든다:
   ```
   # 유역 관측소 목록 (1=한강 2=낙동강 3=금강 4=섬진강 5=영산강)
   python3 scripts/fetch_wamis_levels.py --basin 5 --list --match "나주|담양"
   # 후보 지점 수집 → 2015~2024 일 커버리지 85% 이상만 유지 → series.json
   python3 scripts/fetch_wamis_levels.py --spec regions/<id>/gauge/wamis_spec.json \
       --start 20150101 --end 20241231 --min-coverage 0.85 --out regions/<id>/gauge/series.json
   # 기상(ASOS 일자료) — 로컬에 KMA 키가 없으면 배포 API에서 실측(source=asos)만 받아 저장
   python3 scripts/fetch_region_weather.py --region-id <id> --station-id 156 \
       --out regions/<id>/weather/asos_daily.json
   ```
   지점 좌표·주소·하천명은 `python3 scripts/fetch_wamis_station_info.py --region <id>` 로 채운다.
   `wamis_spec.json` 은 `{region_id, stations:[{id,name,code,terrain,terrain_note}]}` 이다. WAMIS 서버는
   응답을 자주 끊으므로 스크립트가 연 단위 재시도·부분 응답 회수를 하며, 끝내 실패한 해는
   `[실패]` 로 따로 보고한다(결측과 구분).
3. `POST /api/v1/regions/{region_id}/gauge/train` 으로 학습·평가한다 (수 초~수십 초).
   내장 지역은 `scripts/train_gauge_models.py --region <id> --plugins ...` 로 학습해 산출물을 커밋한다.
4. 위성 장면이 있으면 `data/incoming/handover/05_l1_pre/<sensor>_pre/<yyyy>/<yyyymmdd>/`
   규약으로 전처리 장면을 두고 `/api/v1/wbms/segmentation` 또는
   `scripts/wbms_chain_runner.py`로 탐지한다. (WBMS 모델은 테스트베드별 DEM·3-class
   마스크·lsmap 보조자료가 필요하다 — 현재 인도된 것은 `busan`·`daecheong`.)

## 지역별 데이터 현황 (2026-09-09 기준)

| 지역 id | 위성 장면 | GT 라벨 | 탐지 산출(WB 마스크) | 위성 수위 산출 | 수위계 시계열 | 예측 모델 | 실증 가능 범위 |
|---|---|---|---|---|---|---|---|
| `nakdong_busan` 낙동강 하구 | **ICEYE SAR 4** (2020-03-02·03-30·04-15·04-16) + **PlanetScope 4** (02-18·03-12·03-25·04-14) | SAR 4 + 광학 4 (수체 이진) | 8씬 (`data/wbms_runs/wb`), ICEYE IoU 0.925/0.947/0.962/0.881, 광학 0.932~0.948 | 4지점(jeongcheon·hupo·gimhae·gupo) × 8시점 + 융합 LSTM 당일 보정 | WAMIS 6지점 2015~2024 (구포대교=gupo, 정천교=jeongcheon 위성 연결) | 6지점 × D+1..30, 시험 2024 34/36셀 우수 | 탐지 정확도·위성 수위·수위계 대조·수위 예측 **전부** |
| `geum_daecheong` 금강 대청댐 | **PlanetScope 10** (2019-03-23·03-29·04-03·04-04·04-27, 2022-05-14·05-16·05-17·05-22·05-23) | 없음 | 인도 증적만(마스크 파일 미동봉) | 3지점(daecheong·jojeongji·bangsuro) × 10시점, 대청댐은 수위대 보정 후 실측 대비 RMSE 0.359 m | WAMIS 3지점 2019~2024 (3지점 모두 위성 연결) | 3지점 × D+1..30, 시험 2024 18/18셀 우수 | 위성 수위·수위계 대조·수위 예측 (탐지 **정확도**는 라벨이 없어 검증 불가) |
| `han_river` 한강 (대조군) | 없음 | — | — | — | WAMIS 5지점 2015~2024 | 5지점 × D+1..30, 시험 2024 30/30셀 우수 | 수위 예측만 — "시계열만 있어도 모델이 만들어진다" 시연용 |
| `yeongsan_river` 영산강 (담양·광주·나주) | 없음 — 위성 장면이 등록되면 탐지·대조 기능이 열린다 | — | — | — | WAMIS 3지점 2015~2024 — 담양댐 저수지(댐)·극락교(본류 상류·광주)·나주대교(본류 중류); 일 커버리지 담양댐 100.0%, 극락교 100.0%, 나주대교 100.0%; 기상 ASOS 광주 156 | 3지점 × D+1..30, 시험 2024 20/21셀 우수 | 수위 예측만 (지상 수위계 + ASOS 기상) |
| `seomjin_river` 섬진강 (섬진강댐·구례·하동) | 없음 — 위성 장면이 등록되면 탐지·대조 기능이 열린다 | — | — | — | WAMIS 3지점 2015~2024 — 회문리(섬진강댐 직하류)·구례교(본류 중류)·읍내리(본류 하류·하동 감조); 일 커버리지 회문리 98.5%, 구례교 100.0%, 읍내리 100.0%; 기상 ASOS 순천 174 | 3지점 × D+1..30, 시험 2024 20/21셀 우수 | 수위 예측만 (지상 수위계 + ASOS 기상) |
| `geum_mainstem` 금강 본류 (세종·공주·청주) | 없음 — 위성 장면이 등록되면 탐지·대조 기능이 열린다 | — | — | — | WAMIS 3지점 2015~2024 — 금남교(대청댐 하류·세종)·금강교(본류·공주)·미호강교(미호강 지류·청주); 일 커버리지 금남교 99.2%, 금강교 100.0%, 미호강교 97.5%; 기상 ASOS 대전 133 | 3지점 × D+1..30, 시험 2024 21/21셀 우수 | 수위 예측만 (지상 수위계 + ASOS 기상) |
| `nakdong_upper` 낙동강 상류 (안동·상주·구미) | 없음 — 위성 장면이 등록되면 탐지·대조 기능이 열린다 | — | — | — | WAMIS 3지점 2015~2024 — 구담교(안동댐·임하댐 하류 본류)·상주보하(상주보 직하류)·구미대교(본류·구미); 일 커버리지 구담교 99.4%, 상주보하 93.9%, 구미대교 100.0%; 기상 ASOS 안동 136 | 3지점 × D+1..30, 시험 2024 20/21셀 우수 | 수위 예측만 (지상 수위계 + ASOS 기상) |

위성 장면이 없는 지역(`satellite.available=false`)은 수위 예측만 열린다. 위성 장면이 등록되면
(`region.json` `satellite` 절 + 전처리 GeoTIFF) 탐지·대조 기능이 열린다.

2026-09-09 추가된 남부 4개 유역(영산강·섬진강·금강 본류·낙동강 상류)의 자료 주의점
(각 `gauge/wamis_spec.json` 의 `data_notes_ko` 에도 기록):

- 여러 지점에서 **2017~2018년에 수위 기준면이 바뀐다**(예: 읍내리 +4 m, 구례교·극락교 +1 m, 나주대교 +2 m —
  WAMIS 기준면 재설정; 금강교·금남교는 2018년 공주보·세종보 개방으로 실제 저하). 이 변화는 학습 구간
  (`train_end` 2018-12-31) 안에 있고, 검증(2019~2023)·시험(2024)은 바뀐 뒤 구간이라 보고 수치는 일관된 기준면에서 나온다.
  `evaluation.json` 행의 `regime_shift_flag` 로 확인한다.
- 상주보하 2016년 2~7월에 계기 오류로 보이는 값(0 / 35.55 m 등 정상 40 m 아래 약 130일)이 있다. 레지스트리 QC(`quality_control_rows`)는
  0 m 정체(19일)만 제거하고 35.55 m 정체는 정상 범위 규칙 안이라 남는다 — 학습 구간(2016)에만 있고 검증·시험 구간에는 없다.
- 제외한 후보: 안동대교(2019-06-23 4167 m 스파이크 + 2020년 전체 0.03 m 정체), 호국의다리·낙단교(2020년 약 +6 m
  기준면 변경), 칠곡보하·성주대교(2017년 기준면 변경), 섬진강댐·안동댐·임하댐·안동조정지댐(2015~2018 결측으로 커버리지 55~65%),
  송정리·논산대교(커버리지 <85%).

위성 원본: 부산 ICEYE SLC 5개(저장소 루트 `*.h5`, `04_raw/iceye_busan`)·PlanetScope 부산 4주문·대청 2주문(1B 비정사).
SAR 전처리(①)는 GAMMA(상용)가 필요해 인수 전처리본(`05_l1_pre`)을 쓴다. 대청은 WBMS ②③만
지원(①④⑤ `TESTBED_UNSUPPORTED`). 새 지역의 탐지를 돌리려면 카탈로그 등록 + DEM·3-class
마스크(SAR)·AOI 보조자료가 필요하다.
