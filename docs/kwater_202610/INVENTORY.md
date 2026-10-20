# K-water(1세부) 수신 자료 인벤토리 — 2026-10 (kwater_202610)

작성 2026-10-08. 원본 위치는 저장소 루트에 있던 5개 파일(.nitf 3개, .zip 2개)이며, 지금은 모두 아래로 옮겼다(`data/`는 .gitignore 대상).
도구: rasterio 1.5.1 (GDAL 3.12.4). `gdalinfo` CLI와 `sarpy`는 **설치되어 있지 않다**. SICD는 rasterio(GDAL NITF 드라이버)와 XML 직접 파싱으로 읽었다.

## 1. 디렉터리 구성

```
data/incoming/kwater_202610/
├── umbra/                                   # SICD 원본 3개(복소 SAR) + 추출한 SICD XML
│   ├── 2026-09-25-03-03-56_UMBRA-10_SICD_MM-001.nitf (+ .sicd.xml)
│   ├── 2026-09-28-03-10-27_UMBRA-09_SICD_MM-002.nitf (+ .sicd.xml)
│   └── 2026-10-03-03-42-12_UMBRA-10_SICD_MM-003.nitf (+ .sicd.xml)
└── planet/
    ├── drive-download-20261007T142539Z-1-004/   (zip 원본 포함)
    │   ├── 260920_PlanetScope/  260927_PlanetScope/  261002_PlanetScope/   # 내부 zip → 같은 이름 폴더로 풀어 둠
    │   └── 261003_Umbra/        # GEC GeoTIFF + SIDD + PNG + STAC (zip 아님)
    └── drive-download-20261007T142539Z-1-005/   (zip 원본 포함)
        └── 260925_Umbra/  260928_Umbra/   # 내부 zip → 풀어 둠
```

주의: 이름은 "planet"이지만 **두 zip에는 Umbra 파생 산출물(GEC GeoTIFF·SIDD·PNG·STAC) 3세트도 들어 있다**. 004 zip 안의 1003 Umbra만 내부 zip이 아니라 낱개 파일이다. 각 내부 zip(`*_대청댐_*.zip`)은 그대로 두고, 같은 이름의 폴더에 풀었다.

## 2. 파일 목록 (경로는 kwater_202610/ 기준)

### 2.1 Umbra SICD (umbra/)
| 파일 | 크기(B) | file 판정 |
|---|---:|---|
| umbra/2026-09-25-03-03-56_UMBRA-10_SICD_MM-001.nitf | 6,468,712,346 | NITF 2.10 |
| umbra/2026-09-28-03-10-27_UMBRA-09_SICD_MM-002.nitf | 7,155,508,351 | NITF 2.10 |
| umbra/2026-10-03-03-42-12_UMBRA-10_SICD_MM-003.nitf | 6,377,671,921 | NITF 2.10 |
| umbra/*.nitf.sicd.xml (3개, 추출본) | 30,835~30,865 | XML (urn:SICD:1.3.0) |

### 2.2 zip 004 (planet/drive-download-20261007T142539Z-1-004/)
| 파일 | 크기(B) | file 판정 |
|---|---:|---|
| drive-download-…-004.zip | 1,697,882,649 | Zip |
| 260920_PlanetScope/260920_대청댐_psscene_analytic_8b_udm2.zip | 504,855,601 | Zip |
| ↳ composite.tif | 501,952,887 | BigTIFF LE |
| ↳ composite_udm2.tif | 2,750,224 | BigTIFF LE |
| ↳ composite_metadata.json / manifest.json / catalog.json | 27,124 / 3,319 / 260 | JSON |
| ↳ 20260920_{025225_95,025227_93,025229_92}_2530_metadata.json + _3B_AnalyticMS_8b_metadata_clip.xml | 3.8–16 KB | JSON / XML |
| ↳ 20260920_{032134_84,032136_76}_24ae_metadata.json + _3B_AnalyticMS_8b_metadata_clip.xml | 7.5–13 KB | JSON / XML |
| 260927_PlanetScope/260927_대청댐_psscene_analytic_8b_sr_udm2.zip | 396,570,929 | Zip |
| ↳ composite.tif | 392,497,323 | BigTIFF LE |
| ↳ composite_udm2.tif | 3,972,198 | BigTIFF LE |
| ↳ composite_metadata.json / manifest.json / catalog.json | 26,001 / 2,345 / 260 | JSON |
| ↳ 20260927_{025703_40,025705_59,025707_78}_24e9_metadata.json + _3B_…xml | 3.5–19.5 KB | JSON / XML |
| 261002_PlanetScope/261002_대청댐_psscene_analytic_8b_sr_udm2.zip | 393,738,398 | Zip |
| ↳ composite.tif | 390,987,700 | BigTIFF LE |
| ↳ composite_udm2.tif | 2,648,808 | BigTIFF LE |
| ↳ composite_metadata.json / manifest.json / catalog.json | 25,737 / 2,345 / 260 | JSON |
| ↳ 20261002_{033247_38,033249_22,033251_07}_24d1_metadata.json + _3B_…xml | 3.8–19.5 KB | JSON / XML |
| 261003_Umbra/2026-10-03-03-42-12_UMBRA-10_MM.tif | 259,378,763 | TIFF LZW, 8bit (COG) |
| 261003_Umbra/2026-10-03-03-42-12_UMBRA-10_SIDD_MM.nitf | 256,062,687 | NITF |
| 261003_Umbra/2026-10-03-03-42-12_UMBRA-10_MM.png | 846,532 | PNG 1024×1024 gray+alpha |
| 261003_Umbra/2026-10-03-03-42-12_UMBRA-10.stac.v2.json | 6,009 | JSON |

### 2.3 zip 005 (planet/drive-download-20261007T142539Z-1-005/)
| 파일 | 크기(B) | file 판정 |
|---|---:|---|
| drive-download-…-005.zip | 802,391,994 | Zip |
| 260925_Umbra/260925_대청댐_Umbra_0.5m.zip | 396,215,147 | Zip |
| ↳ 2026-09-25-03-03-56_UMBRA-10_MM.tif | 254,233,834 | TIFF LZW, 8bit |
| ↳ 2026-09-25-03-03-56_UMBRA-10_SIDD_MM.nitf | 256,062,680 | NITF |
| ↳ …_MM.png / .stac.v2.json | 847,823 / 6,016 | PNG / JSON |
| 260928_Umbra/260928_대청댐_Umbra_0.5m.zip | 406,053,833 | Zip |
| ↳ 2026-09-28-03-10-27_UMBRA-09_MM.tif | 260,618,639 | TIFF LZW, 8bit |
| ↳ 2026-09-28-03-10-27_UMBRA-09_SIDD_MM.nitf | 256,062,689 | NITF |
| ↳ …_MM.png / .stac.v2.json | 833,320 / 6,011 | PNG / JSON |

## 3. PlanetScope (광학)

공통: item_type **PSScene**, 장비 PSB.SD(SuperDove), **L3B(ortho, 3B)**, 8밴드, quality standard, ground_control true.
배포 형태: **스트립 클립 모자이크 `composite.tif` 1장 + `composite_udm2.tif`**만 들어 있다. 개별 장면 TIF는 없고, 장면별 metadata.json/xml만 있다.
밴드(PSScene 8b 표준 순서): 1 Coastal Blue, 2 Blue, 3 Green I, 4 Green, 5 Yellow, 6 Red, 7 Red Edge, 8 NIR. composite.tif에는 밴드 설명 태그가 없다.

| 날짜 | 번들(product) | 구성 장면(item id) | 촬영 UTC | 위성 | GSD | 운량/청천% | 비고 |
|---|---|---|---|---|---|---|---|
| 2026-09-20 | **analytic_8b_udm2 (SR 아님, TOA 복사휘도)** | 20260920_025225_95_2530, _025227_93_2530, _025229_92_2530, 20260920_032134_84_24ae, _032136_76_24ae | 02:52:26–29 / 03:21:35–37 | 2530, 24ae (2기 혼합) | 3.6 / 3.3 | 0–1 / 99–100 | DN 중앙값 b1≈5300, b8≈6400. 반사율이 아니다 |
| 2026-09-27 | **analytic_8b_sr_udm2** (지표반사율 ×10000) | 20260927_025703_40_24e9, _025705_59_24e9, _025707_78_24e9 | 02:57:03–08 | 24e9 | 3.7 | 2–4 / 94–96 | b8(NIR) 중앙값 2433 |
| 2026-10-02 | **analytic_8b_sr_udm2** (지표반사율 ×10000) | 20261002_033247_38_24d1, _033249_22_24d1, _033251_07_24d1 | 03:32:47–51 | 24d1 | 3.2 | 0 / 100 | b8 중앙값 2387 |

- 동봉된 장면별 XML은 `_3B_AnalyticMS_8b_metadata_clip.xml`(TOA 메타)이라, SR 번들에서도 `atmosphericCorrectionApplied=false`로 적혀 있다. SR 여부는 번들 이름과 화소값 범위(0927·1002는 수백~수천 수준)로 판단했다.
- 0920 번들은 SR이 아니다. 다른 날짜와 같은 척도로 비교하거나 지수(NDWI 등)를 계산하려면 보정을 따로 해야 한다.

### 3.1 래스터 속성 (rasterio)
| 파일 | CRS | 해상도 | W×H | dtype | 밴드 | nodata | 투영 bounds (m) | 경위도 bounds (W,S,E,N) |
|---|---|---|---|---|---|---|---|---|
| 0920 composite.tif | EPSG:32652 (UTM 52N) | 3.0 m | 7170×7683 | uint16 | 8 | 0 | 357867, 4020198, 379377, 4043247 | 127.41247, 36.31633, 127.65624, 36.52700 |
| 0927 composite.tif | EPSG:32652 | 3.0 m | 7170×7683 | uint16 | 8 | 0 | 동일 | 동일 |
| 1002 composite.tif | EPSG:32652 | 3.0 m | 7170×7683 | uint16 | 8 | 0 | 동일 | 동일 |
| */composite_udm2.tif (3개) | EPSG:32652 | 3.0 m | 7170×7683 | uint8 | 8 | 없음 | 동일 | 동일 |

- 세 날짜 모두 **같은 격자**(픽셀 단위 정합)이다. 사각형 범위 중 유효 화소는 약 60.5–61.5%로, 클립 다각형 바깥은 0이다.
- **UDM2가 있다**. 밴드는 clear, snow, shadow, haze_light, haze_heavy, cloud, confidence, udm1 순이다. 유효 영역 대비 cloud 비율은 0920 0.2%, 0927 2.1%(shadow 0.7, haze_light 0.8), 1002 0%이다.
- 유효 영역 실제 범위(composite_metadata.json의 geometry 기준): lon 127.4136–127.656, lat 36.318–36.525.

## 4. Umbra SAR

### 4.1 SICD 원본 (umbra/*.nitf)
rasterio로 열린다. driver NITF, 1밴드 **complex64**, CRS 없음(GCP 4점), 메타 도메인은 NITF_METADATA, xml:DES, TEXT 등이다. `gdalinfo`는 없고, `sarpy`도 설치되어 있지 않다.

NITF 헤더 텍스트(처음 100 KB 안의 ASCII 구간은 2개뿐이고, 나머지는 바이너리 화소다):
| 파일 | FTITLE | ISORCE | IDATIM | ICAT/IREP | IGEOLO(4모서리) | rasterio W×H |
|---|---|---|---|---|---|---|
| 001 | 25SEP26UM10010002026-09-25-03-03-56_UMBRA-10_SICDF | UMBRA-10 | 20260925030356 | SAR/NODISPLY | 362856N1272746E 362516N1272820E 362518N1272416E 362900N1272343E | 60750×13310 |
| 002 | 28SEP26UM09010002026-09-28-03-10-27_UMBRA-09_SICDF | UMBRA-09 | 20260928031027 | SAR/NODISPLY | 362901N1272738E 362500N1272834E 362513N1272425E 362916N1272330E | 67200×13310 |
| 003 | 03OCT26UM10010002026-10-03-03-42-12_UMBRA-10_SICDF | UMBRA-10 | 20261003034212 | SAR/NODISPLY | 362559N1272304E 362917N1272500E 362816N1272852E 362457N1272655E | 59895×13310 |

### 4.2 SICD XML (DES에서 추출, urn:SICD:1.3.0)
| 항목 | 001 (09-25) | 002 (09-28) | 003 (10-03) |
|---|---|---|---|
| CollectorName / CoreName | Umbra-10 / 2026-09-25-03-03-56_Umbra-10 | Umbra-09 / 2026-09-28-03-10-27_Umbra-09 | Umbra-10 / 2026-10-03-03-42-12_Umbra-10 |
| Mode | SPOTLIGHT, MONOSTATIC | SPOTLIGHT | SPOTLIGHT |
| NumRows × NumCols | 13310 × 60750 | 13310 × 67200 | 13310 × 59895 |
| PixelType | RE32F_IM32F | RE32F_IM32F | RE32F_IM32F |
| Grid | SLANT / RGAZIM | SLANT / RGAZIM | SLANT / RGAZIM |
| Row SS / Col SS (m) | 0.337 / 0.111 | 0.263 / 0.108 | 0.266 / 0.112 |
| ImpRespWid Row/Col (m) | 0.373 / 0.122 | 0.291 / 0.119 | 0.295 / 0.124 |
| Polarization Tx / TxRcv | V / V:V | V / V:V | V / V:V |
| TxFrequency (GHz) | 9.405–9.795 (X) | 9.351–9.849 | 9.353–9.847 |
| CollectStart (UTC) / 길이 | 2026-09-25T03:03:56Z / 13.9 s | 2026-09-28T03:10:27Z / 13.1 s | 2026-10-03T03:42:12Z / 11.4 s |
| GrazeAng / IncidenceAng | 41.70° / 48.30° | 54.40° / 35.60° | 53.86° / 36.14° |
| SideOfTrack / AzimAng | **R** / 83.0° | **R** / 79.5° | **L** / 295.3° |
| 궤도(STAC) | descending | descending | descending |
| ImageFormAlgo | PFA | PFA | PFA |
| SCP LLH | 36.452093, 127.433492, 67.6 m | 36.452093, 127.433492, 67.6 m | 36.452000, 127.432999, 70.4 m |
| Radiometric | NoiseLevel(ABSOLUTE), RCSSFPoly, **SigmaZeroSFPoly 있음**, **BetaZeroSFPoly 있음**, GammaZeroSFPoly | 동일 | 동일 |

ImageCorners (lat, lon):
| ICP | 001 | 002 | 003 |
|---|---|---|---|
| 1 FRFC | 36.48221, 127.46285 | 36.48359, 127.46056 | 36.43309, 127.38453 |
| 2 FRLC | 36.42105, 127.47213 | 36.41654, 127.47611 | 36.48801, 127.41674 |
| 3 LRLC | 36.42175, 127.40446 | 36.42034, 127.40692 | 36.47117, 127.48098 |
| 4 LRFC | 36.48326, 127.39518 | 36.48770, 127.39158 | 36.41593, 127.44870 |

- 세 장면 모두 SCP가 같다(36.452N, 127.433E). 같은 지점을 3회 촬영한 것이다.
- 10-03은 **좌측 관측(L)**이고, 나머지 둘은 우측(R)이다. 관측 기하가 반대라 layover·그림자 방향이 다르다. 변화 탐지 시 주의해야 한다.
- 09-25는 입사각 48°이고, 나머지 둘은 약 36°이다. 수면 후방산란 대비가 날짜마다 다를 수 있다.

### 4.3 Umbra 파생 산출물 (zip 안)
| 날짜 | GEC `*_MM.tif` | SIDD `*_SIDD_MM.nitf` | STAC 요약 |
|---|---|---|---|
| 09-25 | EPSG:4326, 16001×16001, uint8, 1밴드, nodata 0, 화소 ≈2.83e-6°×3.48e-6°(≈0.25×0.39 m) | NITF uint8 16001², EPSG:4326 | GEC, VV, X밴드 9.58 GHz, 해상도 거리 0.50 m / 방위 0.125 m, 4 looks(방위) |
| 09-28 | EPSG:4326, 16001², uint8 | NITF uint8 16001², EPSG:4326 | 동일 |
| 10-03 | EPSG:4326, 16001², uint8 | NITF uint8 16001², **rasterio에서 CRS가 안 잡힘** | 동일 |
GEC/SIDD는 8bit 표시용 진폭(디스플레이 스케일)이라, σ0 정량 분석에는 SICD와 SigmaZeroSFPoly를 써야 한다.

## 5. 촬영 범위 (경위도) 및 중첩

| 장면 | lon W–E | lat S–N | 출처 |
|---|---|---|---|
| Planet 0920/0927/1002 (유효영역) | 127.4136–127.656 | 36.318–36.525 | composite_metadata geometry |
| Umbra 09-25 SICD | 127.3952–127.4721 | 36.4210–36.4833 | SICD ImageCorners |
| Umbra 09-28 SICD | 127.3916–127.4761 | 36.4165–36.4877 | SICD ImageCorners |
| Umbra 10-03 SICD | 127.3845–127.4810 | 36.4159–36.4880 | SICD ImageCorners |
| Umbra 09-25 GEC | 127.4024–127.4646 | 36.4270–36.4772 | GeoTIFF/STAC bbox |
| Umbra 09-28 GEC | 127.4010–127.4660 | 36.4258–36.4784 | GeoTIFF |
| Umbra 10-03 GEC | 127.3959–127.4701 | 36.4220–36.4820 | GeoTIFF |

중첩(다각형 면적 기준, shapely):
| 비교 | 결과 |
|---|---|
| 지정 영역(lon 127.45–127.55, lat 36.40–36.52) ⊂ Planet | 지정 영역의 **85%**를 Planet이 덮는다(세 날짜 동일) |
| 지정 영역 ⊂ Umbra SICD | 지정 영역의 **8–10%**만 덮는다. SAR 장면 면적의 23–26%만 지정 영역 안에 있다 |
| 지정 영역 ⊂ Umbra GEC | 지정 영역의 4%만 덮는다 |
| Umbra GEC ⊂ Planet | GEC 면적의 82–84%가 Planet 안에 있다. Planet 서쪽 경계(127.4136)보다 서쪽은 빠진다 |
| Umbra SICD ⊂ Planet | SICD 면적의 62–66%가 Planet 안에 있다 |
| Umbra 상호 (GEC) | 09-25↔09-28 97%, 10-03↔09-25 84%, 10-03↔09-28 83% |

해석:
- SAR 3장면은 SCP(127.433E, 36.452N)를 중심으로 약 6×6 km(GEC) 범위다. 지정한 상자(127.45–127.55)보다 **서쪽에 치우쳐** 있어, 상자와는 동쪽 끝(127.45–127.48)만 겹친다. 대청댐 본체가 대략 127.48E, 36.47N 부근(대략값, 지도로 확인 필요)이라면, SAR 장면은 댐 바로 아래에서 서쪽(신탄진 방향)으로 흐르는 금강 하류 구간을 겨냥한 것으로 보인다. 그렇다면 "직하류" 확인용으로는 오히려 맞는 위치이고, 지정 상자 쪽이 동쪽(저수지 쪽)으로 치우쳤을 가능성이 있다. 퀵룩 PNG나 지도로 확인할 것을 권한다.
- 광학과 SAR의 공통 영역은 대략 lon 127.414–127.47, lat 36.42–36.48이다. 이 범위에서 SAR(09-25, 09-28, 10-03)와 Planet(09-20, 09-27, 10-02)을 짝지을 수 있다. 날짜 차이가 가장 작은 쌍은 09-28 SAR ↔ 09-27 Planet SR(1일), 10-03 SAR ↔ 10-02 Planet SR(1일)이다.

## 6. 미확인·주의 사항
- `gdalinfo`와 `sarpy`가 없다. 설치하지 않았다. SICD 복소 화소는 rasterio(GDAL NITF)로 열리는 것까지만 확인했고, 화소 판독과 σ0 변환은 하지 않았다.
- Planet 0920은 SR이 아니라 analytic(TOA)이다.
- Planet은 개별 장면 GeoTIFF가 아니라 composite만 들어 있다(장면 경계는 장면별 metadata.json geometry에 있다).
- 10-03 SIDD NITF는 rasterio에서 CRS가 비어 있다. 같은 날짜의 GEC GeoTIFF(`_MM.tif`)는 EPSG:4326으로 정상이다.
- 저장소 루트에는 이번 5개와 무관한 이전 zip(`drive-download-20260820T073437Z-1-0xx.zip` 14개)이 남아 있다. 건드리지 않았다.
