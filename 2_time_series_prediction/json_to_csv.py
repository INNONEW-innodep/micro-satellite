"""
기상청 API JSON 응답을 CSV 형식으로 변환하는 스크립트
"""

import json
import csv
import argparse
from datetime import datetime
import os


def parse_date(date_str):
    """
    날짜 문자열을 CSV 형식으로 변환
    예: "2020-02-18" → "2020. 2. 18"
    """
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return f"{dt.year}. {dt.month}. {dt.day}"
    except:
        return date_str


def safe_float(value, default=0.0):
    """
    안전하게 float로 변환 (빈 문자열이나 None 처리)
    """
    if value == "" or value is None:
        return default
    try:
        return float(value)
    except:
        return default


def json_to_csv(json_path: str, csv_path: str):
    """
    기상청 API JSON 파일을 CSV 형식으로 변환
    
    Parameters
    ----------
    json_path : str
        입력 JSON 파일 경로
    csv_path : str
        출력 CSV 파일 경로
    """
    # JSON 파일 로드
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 데이터 추출
    try:
        items = data['response']['body']['items']['item']
    except KeyError as e:
        print(f"❌ JSON 구조 오류: {e}")
        print("   'response.body.items.item' 경로를 확인하세요.")
        return False
    
    if not items:
        print("❌ 데이터가 없습니다.")
        return False
    
    # CSV 작성
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        
        # 헤더 작성
        writer.writerow([
            '일자(date)',
            '간격 (Δt)',
            'humidity (%)',
            'precipitation (mm)',
            'tmp_min (°C)',
            'tmp_max (°C)',
            'pressure (hPa)',
            'wind_speed (m/s)'
        ])
        
        # 데이터 행 작성
        for item in items:
            date_str = parse_date(item.get('tm', ''))
            humidity = safe_float(item.get('avgRhm', ''))
            precipitation = safe_float(item.get('sumRn', ''))
            tmp_min = safe_float(item.get('minTa', ''))
            tmp_max = safe_float(item.get('maxTa', ''))
            pressure = safe_float(item.get('avgPa', ''))
            wind_speed = safe_float(item.get('avgWs', ''))
            
            writer.writerow([
                date_str,
                '',  # 간격은 비워둠 (나중에 계산 가능)
                humidity,
                precipitation,
                tmp_min,
                tmp_max,
                pressure,
                wind_speed
            ])
    
    print(f"✓ CSV 변환 완료: {csv_path}")
    print(f"  총 {len(items)}일 데이터 변환됨")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="기상청 API JSON 응답을 CSV로 변환"
    )
    parser.add_argument(
        '--json_path',
        type=str,
        required=True,
        help='입력 JSON 파일 경로'
    )
    parser.add_argument(
        '--csv_path',
        type=str,
        default=None,
        help='출력 CSV 파일 경로 (기본값: JSON 파일과 같은 디렉토리)'
    )
    
    args = parser.parse_args()
    
    # CSV 경로가 지정되지 않으면 JSON 파일과 같은 디렉토리에 저장
    if args.csv_path is None:
        base_name = os.path.splitext(os.path.basename(args.json_path))[0]
        csv_dir = os.path.dirname(args.json_path)
        args.csv_path = os.path.join(csv_dir, f"{base_name}.csv")
    
    # 변환 실행
    json_to_csv(args.json_path, args.csv_path)


if __name__ == "__main__":
    main()

