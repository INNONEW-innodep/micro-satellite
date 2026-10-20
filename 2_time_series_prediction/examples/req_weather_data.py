"""
기상청 일자료 조회서비스 API 기반
기상 과거 관측 데이터 조회 모듈
https://www.data.go.kr/data/15059093/openapi.do
"""

import requests
import json
import os

url = 'http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList'
# 인증키는 코드에 적지 않는다. 과거 커밋에 평문으로 들어간 키는 폐기·재발급 대상이다.
service_key = os.environ.get("KMA_API_KEY", "")
if not service_key:
    raise SystemExit("환경변수 KMA_API_KEY 에 공공데이터포털 일반 인증키를 설정하세요 (예: export KMA_API_KEY=...)")

# YYYYMMDD format (D-1까지 제공)
start_date="20250730"
end_date="20250730"

params ={
    'serviceKey' : service_key,
    'pageNo' : '1',
    'numOfRows' : '10',
    'dataType' : 'JSON',
    'dataCd' : 'ASOS',
    'dateCd' : 'DAY',
    'startDt' : start_date,
    'endDt' : end_date, 
    'stnIds' : '108' }

response = requests.get(url, params=params)
data = response.json()

save_dir = "./data"
os.makedirs(save_dir, exist_ok=True)
f_path = os.path.join(save_dir, f"weather_{start_date}_{end_date}.json")

with open(f_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"Saved weather data: {f_path}")