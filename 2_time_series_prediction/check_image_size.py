"""
원본 이미지 크기 확인 스크립트
"""
import numpy as np
import rasterio
import os

print("=" * 60)
print("원본 래스터 파일 크기 확인")
print("=" * 60)

raster_dir = "data/busan/raster"
raster_files = sorted([f for f in os.listdir(raster_dir) if f.endswith('.tif')])

for f in raster_files:
    raster_path = os.path.join(raster_dir, f)
    with rasterio.open(raster_path) as src:
        print(f"\n파일: {f}")
        print(f"  원본 크기: {src.width} x {src.height} (width x height)")
        print(f"  CRS: {src.crs}")
        print(f"  Bounds: {src.bounds}")

print("\n" + "=" * 60)
print("data.npy 패치 정보 확인")
print("=" * 60)

data = np.load("data/busan/data.npy")
print(f"\nDataset shape: {data.shape}")
print(f"  패치 수: {data.shape[0]}")
print(f"  프레임 수: {data.shape[1]}")
print(f"  패치 크기: {data.shape[2]} x {data.shape[3]}")

# 패치 그리드 계산
n_patches = data.shape[0]
patch_size = data.shape[2]

import math
n_per_side = int(math.ceil(math.sqrt(n_patches)))
print(f"\n패치 그리드 계산:")
print(f"  패치 그리드: {n_per_side} x {n_per_side} = {n_per_side*n_per_side} (총 {n_patches}개 패치)")
print(f"  추정 원본 크기: {n_per_side * patch_size} x {n_per_side * patch_size}")

# 실제 원본 크기와 비교
if raster_files:
    with rasterio.open(os.path.join(raster_dir, raster_files[0])) as src:
        actual_width = src.width
        actual_height = src.height
        print(f"\n실제 원본 크기: {actual_width} x {actual_height}")
        print(f"추정 크기와 비교: {n_per_side * patch_size} x {n_per_side * patch_size}")
        
        # 패딩 계산
        padded_width = n_per_side * patch_size
        padded_height = n_per_side * patch_size
        pad_w = padded_width - actual_width
        pad_h = padded_height - actual_height
        print(f"\n패딩 정보:")
        print(f"  패딩된 크기: {padded_width} x {padded_height}")
        print(f"  패딩량: width={pad_w}, height={pad_h}")

