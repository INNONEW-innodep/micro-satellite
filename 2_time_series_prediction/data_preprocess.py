import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import rasterio

def parse_args():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser(description="Preprocess conv-lstm train data")
    parser.add_argument("--raster_dir", type=str, default="./data/raster",
                        help="Input raster (.tif) directory")
    parser.add_argument("--npy_dir", type=str, default="./data/npy",
                        help="Output directory for .npy files")
    parser.add_argument("--img_dir", type=str, default=None,
                        help="If set, save visualization images to this directory")
    parser.add_argument("--save_data_path", type=str, default="./data/data.npy",
                        help="Output file path for preprocessed data")

    args = parser.parse_args()


    return args

def raster_to_np(raster_fpath: str) -> np.ndarray:
    """
    래스터(.tif) 파일을 np array로 변환 후 같은 디렉토리에 .npy 파일로 저장

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

    # 1) raster → npy 변환
    for file_name in sorted(os.listdir(args.raster_dir)):
        if file_name.lower().endswith(".tif"):
            raster_path = os.path.join(args.raster_dir, file_name)
            base_name = os.path.splitext(file_name)[0]
            save_path = os.path.join(args.npy_dir, base_name + ".npy")

            arr = raster_to_np(raster_path)
            np.save(save_path, arr)
            print(f"Converted {file_name} → {base_name}.npy")

    # 2) npy 로드 + 시각화 + 패치 분할
    patch_size = (512, 512)
    patch_sequences = []

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
                plt.colorbar(label='Label (0: background, 1: water)')
                plt.savefig(save_path)
                plt.close()
                print(f"Visualized {base_name}: {base_name}.png")

            # 패치 분할
            patches = split_into_patches_with_padding(arr, patch_size)
            for idx, patch in enumerate(patches):
                if len(patch_sequences) <= idx:
                    patch_sequences.append([])
                patch_sequences[idx].append(patch)

    # 3) dataset 저장
    arr = np.array(patch_sequences)
    data = np.expand_dims(arr, axis=-1)
    print(f"Preprocessing completed. Dataset shape: {data.shape}")

    np.save(args.save_data_path, data)
    print(f"Saved preprocessed dataset to: {args.save_data_path}")