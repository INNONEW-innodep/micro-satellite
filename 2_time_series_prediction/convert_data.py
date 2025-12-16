"""
Convert raster data to numpy array
- 지리적 정렬(Georeferencing) 지원: 공통 영역으로 클리핑
"""


import argparse
import os
import json

import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window, from_bounds

def parse_args():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser(description="Convert raster data to numpy array")
    parser.add_argument("--raster_dir", type=str, default="./data/raster",
                        help="Input raster (.tif) directory")
    parser.add_argument("--npy_dir", type=str, default="./data/npy",
                        help="Output directory for .npy files")
    parser.add_argument("--img_dir", type=str, default=None,
                        help="If set, save visualization images to this directory")
    parser.add_argument("--save_data_path", type=str, default="./data/data.npy",
                        help="Output file path for preprocessed data")
    parser.add_argument("--align_geo", action="store_true",
                        help="Enable geographic alignment (clip to common area)")
    parser.add_argument("--save_geo_info", type=str, default=None,
                        help="Save geographic info (common bounds) to JSON file")

    args = parser.parse_args()

    return args


def get_common_bounds(raster_dir: str) -> dict:
    """
    모든 래스터 파일의 공통 영역(intersection) 계산
    
    Args:
        raster_dir: 래스터 파일 디렉토리
        
    Returns:
        dict: 공통 영역 정보 (bounds, crs, resolution, size)
    """
    raster_files = sorted([f for f in os.listdir(raster_dir) if f.lower().endswith('.tif')])
    
    if not raster_files:
        raise ValueError(f"래스터 파일이 없습니다: {raster_dir}")
    
    all_bounds = []
    all_crs = []
    all_resolutions = []
    
    for f in raster_files:
        raster_path = os.path.join(raster_dir, f)
        with rasterio.open(raster_path) as src:
            all_bounds.append(src.bounds)
            all_crs.append(src.crs)
            all_resolutions.append((abs(src.transform.a), abs(src.transform.e)))
    
    # CRS 일치 확인
    crs_set = set([str(c) for c in all_crs])
    if len(crs_set) > 1:
        raise ValueError(f"CRS가 다릅니다! 동일한 CRS로 변환 후 사용하세요: {crs_set}")
    
    # 해상도 일치 확인 (소수점 6자리까지 비교)
    res_set = set([(round(r[0], 6), round(r[1], 6)) for r in all_resolutions])
    if len(res_set) > 1:
        raise ValueError(f"해상도가 다릅니다! 동일한 해상도로 리샘플링 후 사용하세요: {res_set}")
    
    # 공통 영역 계산 (intersection)
    common_left = max(b.left for b in all_bounds)
    common_right = min(b.right for b in all_bounds)
    common_bottom = max(b.bottom for b in all_bounds)
    common_top = min(b.top for b in all_bounds)
    
    if common_left >= common_right or common_bottom >= common_top:
        raise ValueError("공통 영역이 없습니다! 이미지들이 겹치지 않습니다.")
    
    res_x, res_y = all_resolutions[0]
    common_width_px = int((common_right - common_left) / res_x)
    common_height_px = int((common_top - common_bottom) / res_y)
    
    return {
        'left': common_left,
        'right': common_right,
        'bottom': common_bottom,
        'top': common_top,
        'crs': str(all_crs[0]),
        'resolution': (res_x, res_y),
        'width_px': common_width_px,
        'height_px': common_height_px,
        'files': raster_files
    }


def raster_to_np_aligned(raster_fpath: str, common_bounds: dict) -> np.ndarray:
    """
    래스터 파일을 공통 영역으로 클리핑하여 numpy 배열로 변환
    
    Args:
        raster_fpath: 래스터 파일 경로
        common_bounds: 공통 영역 정보
        
    Returns:
        np.ndarray: 클리핑된 래스터 이미지
    """
    with rasterio.open(raster_fpath) as src:
        # 공통 영역에 해당하는 윈도우 계산
        window = from_bounds(
            common_bounds['left'],
            common_bounds['bottom'],
            common_bounds['right'],
            common_bounds['top'],
            src.transform
        )
        
        # 윈도우 내 데이터 읽기
        array = src.read(1, window=window)
        
        # 크기 확인 및 조정 (경계 효과로 인한 1픽셀 차이 보정)
        expected_height = common_bounds['height_px']
        expected_width = common_bounds['width_px']
        
        if array.shape[0] != expected_height or array.shape[1] != expected_width:
            # 패딩 또는 자르기로 크기 맞춤
            result = np.zeros((expected_height, expected_width), dtype=array.dtype)
            h = min(array.shape[0], expected_height)
            w = min(array.shape[1], expected_width)
            result[:h, :w] = array[:h, :w]
            array = result
        
        return array


def raster_to_np(raster_fpath: str) -> np.ndarray:
    """
    래스터(.tif) 파일을 np array로 변환 (기존 방식, 정렬 없음)

    Args:
        raster_fpath (str): 래스터 이미지 파일 경로

    Returns:
        np.ndarray: 래스터 이미지를 np array로 변환한 결과 값
    """
    with rasterio.open(raster_fpath) as src:
        array = src.read(1)

    return array

def split_into_patches_with_padding(img: np.ndarray, patch_size: tuple):
    """
    이미지를 일정 크기의 패치로 분할하고, 부족한 부분은 0으로 패딩.

    Args:
        img (np.ndarray): 원본 이미지 (2D 또는 3D 가능, 예: (H, W) 또는 (H, W, C))
        patch_size (tuple): (patch_height, patch_width)

    Returns:
        list[np.ndarray]: 패딩 포함 동일 크기의 패치 리스트
    """
    h, w = img.shape[:2]
    ph, pw = patch_size

    # 패딩이 필요한 크기 계산
    pad_h = (ph - (h % ph)) % ph
    pad_w = (pw - (w % pw)) % pw

    # 2D or 3D 대응
    if img.ndim == 2:
        padded_img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    else:
        padded_img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)

    H, W = padded_img.shape[:2]
    patches = []

    for y in range(0, H, ph):
        for x in range(0, W, pw):
            patch = padded_img[y:y+ph, x:x+pw]
            patches.append(patch)

    return patches

if __name__ == "__main__":

    # ./data/raster 하위에 raster(.tif) 파일 적재
    args = parse_args()

    os.makedirs(args.npy_dir, exist_ok=True)
    if args.img_dir:
        os.makedirs(args.img_dir, exist_ok=True)

    # 지리적 정렬 활성화 시 공통 영역 계산
    common_bounds = None
    if args.align_geo:
        print("=" * 60)
        print("지리적 정렬(Geo-alignment) 활성화")
        print("=" * 60)
        
        common_bounds = get_common_bounds(args.raster_dir)
        
        print(f"\n공통 영역 정보:")
        print(f"  CRS: {common_bounds['crs']}")
        print(f"  Bounds (left, bottom, right, top):")
        print(f"    West (left):   {common_bounds['left']:.2f}")
        print(f"    South (bottom): {common_bounds['bottom']:.2f}")
        print(f"    East (right):  {common_bounds['right']:.2f}")
        print(f"    North (top):   {common_bounds['top']:.2f}")
        print(f"  해상도: {common_bounds['resolution'][0]:.2f} x {common_bounds['resolution'][1]:.2f}")
        print(f"  공통 영역 크기: {common_bounds['width_px']} x {common_bounds['height_px']} 픽셀")
        print()
        
        # 지리 정보 저장
        if args.save_geo_info:
            geo_info_path = args.save_geo_info
            with open(geo_info_path, 'w', encoding='utf-8') as f:
                json.dump(common_bounds, f, indent=2, ensure_ascii=False)
            print(f"지리 정보 저장됨: {geo_info_path}\n")

    # 1) raster → npy 변환
    for file_name in sorted(os.listdir(args.raster_dir)):
        if file_name.lower().endswith(".tif"):
            raster_path = os.path.join(args.raster_dir, file_name)
            base_name = os.path.splitext(file_name)[0]
            save_path = os.path.join(args.npy_dir, base_name + ".npy")

            if args.align_geo and common_bounds:
                # 공통 영역으로 클리핑하여 변환
                arr = raster_to_np_aligned(raster_path, common_bounds)
                print(f"Converted (aligned) {file_name} → {base_name}.npy, shape: {arr.shape}")
            else:
                # 기존 방식 (정렬 없음)
                arr = raster_to_np(raster_path)
                print(f"Converted {file_name} → {base_name}.npy")
            
            np.save(save_path, arr)

    # 2) npy 로드 + 시각화 + 패치 분할
    patch_size = (512, 512)
    patch_sequences = []
    all_patches_list = []  # 각 이미지의 패치 리스트를 저장

    # 먼저 모든 이미지를 로드하고 패치로 분할
    for file_name in sorted(os.listdir(args.npy_dir)):
        if file_name.lower().endswith(".npy"):
            file_path = os.path.join(args.npy_dir, file_name)
            arr = np.load(file_path)
            base_name = os.path.splitext(file_name)[0]

            # 시각화
            if args.img_dir:
                save_path = os.path.join(args.img_dir, base_name + ".png")
                plt.imshow(arr, cmap="gray", origin="upper")
                plt.title(f"Water Mask: {base_name}")
                plt.colorbar(label='Value (0: background, 1: water)')
                plt.savefig(save_path)
                plt.close()
                print(f"Visualized {base_name}: {base_name}.png")

            # 패치 분할
            patches = split_into_patches_with_padding(arr, patch_size)
            all_patches_list.append(patches)
            print(f"  {base_name}: {len(patches)} patches, image shape: {arr.shape}")

    # 패치 수 확인 및 통일
    n_patches_per_image = [len(patches) for patches in all_patches_list]
    min_patches = min(n_patches_per_image)
    max_patches = max(n_patches_per_image)
    
    if min_patches != max_patches:
        print(f"\n⚠️  경고: 이미지별 패치 수가 다릅니다. (최소: {min_patches}, 최대: {max_patches})")
        print(f"   최소 패치 수({min_patches})에 맞춰서 자르겠습니다.")
        
        # 최소 패치 수에 맞춰서 자르기
        for i, patches in enumerate(all_patches_list):
            if len(patches) > min_patches:
                all_patches_list[i] = patches[:min_patches]
                print(f"   이미지 {i+1}: {len(patches)} → {min_patches} 패치로 조정")
    else:
        print(f"\n✓ 모든 이미지가 동일한 패치 수를 가집니다: {min_patches}")

    # 패치 시퀀스 재구성: (n_patches, n_frames, patch_h, patch_w)
    n_frames = len(all_patches_list)
    n_patches = min_patches
    
    for patch_idx in range(n_patches):
        patch_sequence = []
        for frame_idx in range(n_frames):
            patch_sequence.append(all_patches_list[frame_idx][patch_idx])
        patch_sequences.append(patch_sequence)

    # 3) dataset 저장
    arr = np.array(patch_sequences)
    data = np.expand_dims(arr, axis=-1)
    print(f"\nPreprocessing completed. Dataset shape: {data.shape}")
    print(f"  (n_patches={data.shape[0]}, n_frames={data.shape[1]}, patch_size={data.shape[2]}x{data.shape[3]})")

    np.save(args.save_data_path, data)
    print(f"Saved preprocessed dataset to: {args.save_data_path}")