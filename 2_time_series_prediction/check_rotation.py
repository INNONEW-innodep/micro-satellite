"""
래스터 이미지의 회전/변환 정보 확인 스크립트
- 각 이미지의 transform 행렬 분석
- 회전 각도 추정
"""
import os
import math
import numpy as np
import rasterio


def analyze_transform(raster_dir: str):
    """래스터 파일들의 변환 행렬 분석"""
    
    raster_files = sorted([f for f in os.listdir(raster_dir) if f.lower().endswith('.tif')])
    
    if not raster_files:
        print("래스터 파일이 없습니다.")
        return
    
    print("=" * 80)
    print("래스터 파일 변환 행렬(Transform) 분석")
    print("=" * 80)
    
    for f in raster_files:
        raster_path = os.path.join(raster_dir, f)
        with rasterio.open(raster_path) as src:
            transform = src.transform
            
            # Affine 변환 행렬 구성요소
            # | a  b  c |   | scale_x  shear_x  translate_x |
            # | d  e  f | = | shear_y  scale_y  translate_y |
            # | 0  0  1 |   |    0        0          1      |
            
            a = transform.a  # x 스케일 (픽셀당 x 방향 거리)
            b = transform.b  # x-y shear (회전 성분)
            c = transform.c  # x 원점 (left)
            d = transform.d  # y-x shear (회전 성분)
            e = transform.e  # y 스케일 (음수: 위에서 아래로)
            f = transform.f  # y 원점 (top)
            
            print(f"\n파일: {f}")
            print(f"  크기: {src.width} x {src.height}")
            print(f"  Transform 행렬:")
            print(f"    | {a:12.6f}  {b:12.6f}  {c:12.6f} |")
            print(f"    | {d:12.6f}  {e:12.6f}  {f:12.6f} |")
            print(f"    |       0.0              0.0              1.0       |")
            
            # 회전 각도 계산 (shear 성분에서)
            # 회전이 있으면 b와 d가 0이 아님
            if abs(b) > 1e-10 or abs(d) > 1e-10:
                # 회전 각도 추정 (라디안 → 도)
                rotation_rad = math.atan2(d, a)
                rotation_deg = math.degrees(rotation_rad)
                print(f"  ⚠️ 회전 감지:")
                print(f"    Shear X (b): {b:.6f}")
                print(f"    Shear Y (d): {d:.6f}")
                print(f"    추정 회전 각도: {rotation_deg:.4f}°")
            else:
                print(f"  ✓ 회전 없음 (축 정렬됨)")
            
            # 스케일 (해상도)
            scale_x = abs(a)
            scale_y = abs(e)
            print(f"  해상도: {scale_x:.4f} x {scale_y:.4f}")
            
            # 원점 좌표
            print(f"  원점 (top-left): ({c:.2f}, {f:.2f})")
    
    # 첫 번째 이미지 대비 차이 분석
    print("\n" + "=" * 80)
    print("첫 번째 이미지 대비 Transform 차이")
    print("=" * 80)
    
    ref_transform = None
    for i, f in enumerate(raster_files):
        raster_path = os.path.join(raster_dir, f)
        with rasterio.open(raster_path) as src:
            transform = src.transform
            
            if i == 0:
                ref_transform = transform
                print(f"\n[기준] {f}")
                continue
            
            print(f"\n{f}:")
            print(f"  a (scale_x) 차이: {transform.a - ref_transform.a:+.6f}")
            print(f"  b (shear_x) 차이: {transform.b - ref_transform.b:+.6f}")
            print(f"  c (origin_x) 차이: {transform.c - ref_transform.c:+.2f}")
            print(f"  d (shear_y) 차이: {transform.d - ref_transform.d:+.6f}")
            print(f"  e (scale_y) 차이: {transform.e - ref_transform.e:+.6f}")
            print(f"  f (origin_y) 차이: {transform.f - ref_transform.f:+.2f}")


def check_image_orientation(raster_dir: str):
    """이미지 방향 및 정합 필요성 확인"""
    
    raster_files = sorted([f for f in os.listdir(raster_dir) if f.lower().endswith('.tif')])
    
    print("\n" + "=" * 80)
    print("이미지 정합 필요성 분석")
    print("=" * 80)
    
    needs_registration = False
    transforms = []
    
    for f in raster_files:
        raster_path = os.path.join(raster_dir, f)
        with rasterio.open(raster_path) as src:
            transforms.append(src.transform)
    
    # Shear 성분 확인
    shears = [(t.b, t.d) for t in transforms]
    max_shear = max(max(abs(s[0]), abs(s[1])) for s in shears)
    
    if max_shear > 1e-10:
        print(f"\n⚠️ 회전/전단(shear) 성분 감지됨 (max: {max_shear:.6f})")
        needs_registration = True
    
    # 스케일 일치 확인
    scales = [(abs(t.a), abs(t.e)) for t in transforms]
    scale_x_range = max(s[0] for s in scales) - min(s[0] for s in scales)
    scale_y_range = max(s[1] for s in scales) - min(s[1] for s in scales)
    
    if scale_x_range > 0.01 or scale_y_range > 0.01:
        print(f"\n⚠️ 스케일 차이 감지됨")
        print(f"   X 스케일 범위: {scale_x_range:.6f}")
        print(f"   Y 스케일 범위: {scale_y_range:.6f}")
        needs_registration = True
    
    # 원점 차이 확인
    origins = [(t.c, t.f) for t in transforms]
    origin_x_range = max(o[0] for o in origins) - min(o[0] for o in origins)
    origin_y_range = max(o[1] for o in origins) - min(o[1] for o in origins)
    
    print(f"\n원점 차이:")
    print(f"  X 원점 범위: {origin_x_range:.2f} (미터)")
    print(f"  Y 원점 범위: {origin_y_range:.2f} (미터)")
    
    if needs_registration:
        print(f"""
================================================================================
권장 사항: 이미지 정합(Image Registration) 필요
================================================================================

이미지들이 회전되어 있거나 스케일이 다릅니다.
시계열 분석을 위해 다음 중 하나를 수행해야 합니다:

1. GIS 소프트웨어로 사전 정합
   - QGIS, ArcGIS 등에서 이미지를 기준 이미지에 정합
   - 정합된 이미지를 다시 내보내기

2. Python 기반 자동 정합 (OpenCV)
   - 특징점 기반 정합 (ORB, SIFT)
   - Affine 또는 Homography 변환 적용

3. 위성 이미지 처리 라이브러리 사용
   - GDAL/rasterio의 warp 기능
   - 기준 이미지에 맞춰 리샘플링
""")
    else:
        print(f"""
================================================================================
✓ 이미지들이 축 정렬(axis-aligned)되어 있습니다.
================================================================================

회전/전단 성분이 없으므로, 좌표 기반 클리핑(--align_geo)만으로 충분합니다.
""")
    
    return needs_registration


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="래스터 이미지 회전/변환 분석")
    parser.add_argument("--raster_dir", type=str, default="./data/busan/raster",
                        help="래스터 파일 디렉토리")
    
    args = parser.parse_args()
    
    analyze_transform(args.raster_dir)
    check_image_orientation(args.raster_dir)

