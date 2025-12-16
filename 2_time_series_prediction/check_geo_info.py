"""
래스터 파일들의 지리 정보 확인 스크립트
- 각 파일의 범위(bounds), CRS, 해상도 확인
- 공통 영역(intersection) 계산
"""
import os
import rasterio
from rasterio.crs import CRS

def check_raster_geo_info(raster_dir: str):
    """래스터 파일들의 지리 정보 확인"""
    
    raster_files = sorted([f for f in os.listdir(raster_dir) if f.lower().endswith('.tif')])
    
    if not raster_files:
        print("래스터 파일이 없습니다.")
        return
    
    print("=" * 80)
    print("래스터 파일 지리 정보")
    print("=" * 80)
    
    all_bounds = []
    all_crs = []
    all_resolutions = []
    
    for f in raster_files:
        raster_path = os.path.join(raster_dir, f)
        with rasterio.open(raster_path) as src:
            bounds = src.bounds
            crs = src.crs
            transform = src.transform
            
            # 해상도 (픽셀당 미터/도)
            res_x = abs(transform.a)  # x 해상도
            res_y = abs(transform.e)  # y 해상도
            
            print(f"\n파일: {f}")
            print(f"  크기: {src.width} x {src.height} (width x height)")
            print(f"  CRS: {crs}")
            print(f"  Bounds (left, bottom, right, top):")
            print(f"    West (left):   {bounds.left:.6f}")
            print(f"    South (bottom): {bounds.bottom:.6f}")
            print(f"    East (right):  {bounds.right:.6f}")
            print(f"    North (top):   {bounds.top:.6f}")
            print(f"  해상도: {res_x:.6f} x {res_y:.6f}")
            print(f"  범위 너비: {bounds.right - bounds.left:.6f}")
            print(f"  범위 높이: {bounds.top - bounds.bottom:.6f}")
            
            all_bounds.append(bounds)
            all_crs.append(crs)
            all_resolutions.append((res_x, res_y))
    
    # CRS 일치 확인
    print("\n" + "=" * 80)
    print("좌표계(CRS) 비교")
    print("=" * 80)
    
    crs_set = set([str(c) for c in all_crs])
    if len(crs_set) == 1:
        print(f"✓ 모든 파일이 동일한 CRS를 사용합니다: {all_crs[0]}")
    else:
        print("⚠️ CRS가 다릅니다! 변환이 필요합니다.")
        for f, c in zip(raster_files, all_crs):
            print(f"  {f}: {c}")
    
    # 해상도 일치 확인
    print("\n" + "=" * 80)
    print("해상도 비교")
    print("=" * 80)
    
    res_set = set(all_resolutions)
    if len(res_set) == 1:
        print(f"✓ 모든 파일이 동일한 해상도를 사용합니다: {all_resolutions[0]}")
    else:
        print("⚠️ 해상도가 다릅니다! 리샘플링이 필요합니다.")
        for f, r in zip(raster_files, all_resolutions):
            print(f"  {f}: {r}")
    
    # 공통 영역 계산
    print("\n" + "=" * 80)
    print("범위(Bounds) 비교 및 공통 영역")
    print("=" * 80)
    
    # 모든 파일의 범위 비교
    lefts = [b.left for b in all_bounds]
    rights = [b.right for b in all_bounds]
    bottoms = [b.bottom for b in all_bounds]
    tops = [b.top for b in all_bounds]
    
    print(f"\n범위 비교:")
    print(f"  Left (West) 범위:   {min(lefts):.6f} ~ {max(lefts):.6f}")
    print(f"  Right (East) 범위:  {min(rights):.6f} ~ {max(rights):.6f}")
    print(f"  Bottom (South) 범위: {min(bottoms):.6f} ~ {max(bottoms):.6f}")
    print(f"  Top (North) 범위:   {min(tops):.6f} ~ {max(tops):.6f}")
    
    # 공통 영역 (intersection)
    common_left = max(lefts)
    common_right = min(rights)
    common_bottom = max(bottoms)
    common_top = min(tops)
    
    if common_left < common_right and common_bottom < common_top:
        print(f"\n✓ 공통 영역 (모든 이미지가 겹치는 부분):")
        print(f"  West (left):   {common_left:.6f}")
        print(f"  South (bottom): {common_bottom:.6f}")
        print(f"  East (right):  {common_right:.6f}")
        print(f"  North (top):   {common_top:.6f}")
        print(f"  공통 영역 너비: {common_right - common_left:.6f}")
        print(f"  공통 영역 높이: {common_top - common_bottom:.6f}")
        
        # 픽셀 크기 계산 (첫 번째 파일 기준)
        res_x, res_y = all_resolutions[0]
        common_width_px = int((common_right - common_left) / res_x)
        common_height_px = int((common_top - common_bottom) / res_y)
        print(f"\n  공통 영역 픽셀 크기 (예상):")
        print(f"    너비: {common_width_px} 픽셀")
        print(f"    높이: {common_height_px} 픽셀")
    else:
        print("\n⚠️ 공통 영역이 없습니다! 이미지들이 겹치지 않습니다.")
    
    # 범위 차이 분석
    print("\n" + "=" * 80)
    print("각 파일별 범위 차이 (첫 번째 파일 대비)")
    print("=" * 80)
    
    ref_bounds = all_bounds[0]
    for i, (f, b) in enumerate(zip(raster_files, all_bounds)):
        if i == 0:
            print(f"\n[기준] {f}")
            continue
        
        print(f"\n{f}:")
        print(f"  Left 차이:   {b.left - ref_bounds.left:+.6f}")
        print(f"  Right 차이:  {b.right - ref_bounds.right:+.6f}")
        print(f"  Bottom 차이: {b.bottom - ref_bounds.bottom:+.6f}")
        print(f"  Top 차이:    {b.top - ref_bounds.top:+.6f}")
    
    print("\n" + "=" * 80)
    print("권장 사항")
    print("=" * 80)
    
    bounds_differ = (max(lefts) != min(lefts) or max(rights) != min(rights) or
                     max(bottoms) != min(bottoms) or max(tops) != min(tops))
    
    if bounds_differ:
        print("""
⚠️ 이미지들의 범위가 다릅니다!
   시계열 분석을 위해 공통 영역으로 클리핑(crop)하는 것을 권장합니다.
   
   convert_data.py를 수정하여 다음 기능을 추가할 수 있습니다:
   1. 모든 이미지의 공통 영역(intersection) 계산
   2. 각 이미지를 공통 영역으로 클리핑
   3. 동일한 해상도로 리샘플링 (필요시)
   4. 클리핑된 이미지를 패치로 분할
""")
    else:
        print("""
✓ 모든 이미지가 동일한 범위를 가집니다.
  현재 convert_data.py를 그대로 사용해도 됩니다.
""")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="래스터 파일 지리 정보 확인")
    parser.add_argument("--raster_dir", type=str, default="./data/busan/raster",
                        help="래스터 파일 디렉토리")
    
    args = parser.parse_args()
    check_raster_geo_info(args.raster_dir)

