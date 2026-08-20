# 기상청 ASOS 일자료 연동 모듈

백엔드에서 기상청 ASOS **과거 일 관측자료**를 같은 스키마로 조회하기 위한 독립 패키지입니다. 실제 API 공급자와 합성 샘플 공급자는 명시적으로 분리되어 있습니다.

> 중요: 이 모듈이 수집하는 ASOS 일자료는 D-1(KST)까지 제공되는 과거 관측자료이며 미래 기상 예보가 아닙니다. 미래 수위 예측 모델의 과거 보조 입력으로 사용할 수 있지만, 미래 기상 입력이 필요하면 별도 예보 공급자를 연결해야 합니다.

공식 명세: [공공데이터포털 기상청 지상(종관, ASOS) 일자료 조회서비스](https://www.data.go.kr/data/15059093/openapi.do)

## 빠른 사용법

API 키는 함수 인자 또는 환경변수로 전달합니다. 저장소에 키를 기록하지 않습니다.

```bash
export KMA_API_KEY='공공데이터포털에서 발급받은 키'
```

```python
from weather_service import AsosDailyClient

client = AsosDailyClient()  # 기본 환경변수: KMA_API_KEY
series = await client.get_daily_observations(
    start_date="2025-07-01",  # YYYY-MM-DD 또는 YYYYMMDD
    end_date="2025-07-31",
    station="부산 (낙동강 하구)",  # 이름/별칭/ASOS 숫자 ID
)

api_payload = series.to_dict()
model_rows = [record.model_features() for record in series.records]
```

함수 인자 키가 환경변수보다 우선합니다.

```python
client = AsosDailyClient(api_key="...")
```

HTTPS가 기본이며 HTTP fallback은 기본적으로 꺼져 있습니다. 배포 환경에서 HTTPS 연결 문제가 확인된 경우에만 선택적으로 켤 수 있습니다.

```python
client = AsosDailyClient(allow_http_fallback=True)
```

## 샘플 공급자

샘플은 네트워크와 API 키 없이 UI/백엔드 계약을 시험할 때만 사용합니다. 동일 입력은 동일한 합성 값을 만들며, 응답에는 `mode="sample"`, `data_category="synthetic_sample"`, `is_sample=True`, `is_forecast=False`가 포함됩니다.

```python
from weather_service import SampleWeatherProvider

provider = SampleWeatherProvider(seed=20260809)
series = await provider.get_daily_observations("20250701", "20250731", 159)
```

실 API와 샘플은 공통 `WeatherProvider` 프로토콜의 다음 메서드를 구현합니다.

```python
async def get_daily_observations(start_date, end_date, station) -> WeatherSeries
```

## 주요 스키마

- `WeatherSeries`: 요청 구간, 관측소, 관측 레코드, 누락 날짜, 출처 메타데이터
- `DailyWeatherObservation`: 강수량·기온·습도·기압·풍속 등 정규화된 값과 원본 `raw`
- `WeatherMetadata`: live/sample 구분, 관측/예보 구분, 공식 출처, 실제 호출 endpoint
- `Station`: ASOS 지점 ID, 지점명, UI 별칭

`record.model_features()`의 안정된 기본 피처는 다음과 같습니다.

```text
precipitation_mm
avg_temperature_c
min_temperature_c
max_temperature_c
avg_humidity_percent
avg_local_pressure_hpa
avg_wind_speed_m_s
```

업스트림 결측값은 0으로 바꾸지 않고 `None`으로 유지합니다. 모델별 결측 처리와 정규화는 모델 adapter에서 결정해야 합니다. `series.to_dict(include_raw=True)`를 사용하면 원본 API item도 API 응답에 포함할 수 있습니다.

## 내장 지점 매핑

UI의 서울(108), 부산(159), 대구(143), 광주(156)를 포함해 인천(112), 수원(119), 대전(133), 울산(152), 제주(184)를 이름으로 조회할 수 있습니다. 매핑에 없는 최신 지점도 공식 숫자 ID를 직접 전달하면 조회할 수 있습니다.

## 오류

- `WeatherConfigurationError`: API 키 누락
- `WeatherValidationError`: 날짜/지점/설정 오류 또는 D-1 이후 요청
- `WeatherTransportError`: 네트워크·timeout·HTTP 오류
- `WeatherApiError`: 기상청/공공데이터포털 결과 코드 오류
- `WeatherResponseFormatError`: JSON/XML 또는 item 구조 오류

오류 메시지에는 쿼리 URL과 API 키를 포함하지 않습니다.

## 테스트

외부 패키지와 실제 API 호출 없이 표준 라이브러리로 실행됩니다.

```bash
python3 -m unittest discover -s weather_service/tests -v
```

