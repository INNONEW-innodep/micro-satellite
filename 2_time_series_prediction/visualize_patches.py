"""
원본 이미지를 패치로 분할한 결과를 시각화하는 스크립트
- 패치들을 그리드 형태로 배치하여 한 장의 이미지로 저장
- 패치 경계선 표시
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from convert_data import split_into_patches_with_padding


def visualize_patch_grid(img: np.ndarray, patches: list, patch_size: tuple, 
                         save_path: str = None, show_boundaries: bool = True,
                         max_patches_per_row: int = None):
    """
    패치들을 그리드 형태로 시각화
    
    Parameters
    ----------
    img : np.ndarray
        원본 이미지
    patches : list
        패치 리스트
    patch_size : tuple
        패치 크기 (height, width)
    save_path : str, optional
        저장 경로
    show_boundaries : bool
        패치 경계선 표시 여부
    max_patches_per_row : int, optional
        한 행에 표시할 최대 패치 수 (None이면 자동 계산)
    """
    ph, pw = patch_size
    h, w = img.shape[:2]
    
    # 그리드 크기 계산
    n_patches = len(patches)
    if max_patches_per_row is None:
        # 자동 계산: 대략 정사각형에 가깝게
        n_cols = int(np.ceil(np.sqrt(n_patches)))
    else:
        n_cols = min(max_patches_per_row, n_patches)
    
    n_rows = int(np.ceil(n_patches / n_cols))
    
    # 패딩된 이미지 크기 계산
    padded_h = ((h + ph - 1) // ph) * ph
    padded_w = ((w + pw - 1) // pw) * pw
    
    # 전체 시각화 이미지 크기 계산
    # 각 패치를 작게 표시 (원본 크기의 1/4 정도)
    display_patch_size = (ph // 2, pw // 2)
    display_ph, display_pw = display_patch_size
    
    # 경계선 두께
    border_width = 2 if show_boundaries else 0
    
    # 전체 이미지 크기
    total_width = n_cols * (display_pw + border_width) + border_width
    total_height = n_rows * (display_ph + border_width) + border_width
    
    # 빈 이미지 생성
    vis_img = np.ones((total_height, total_width), dtype=np.float32)
    
    # 패치들을 배치
    patch_idx = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if patch_idx >= n_patches:
                break
            
            # 패치 위치 계산
            y_start = row * (display_ph + border_width) + border_width
            y_end = y_start + display_ph
            x_start = col * (display_pw + border_width) + border_width
            x_end = x_start + display_pw
            
            # 패치 가져오기 및 리사이즈
            patch = patches[patch_idx]
            if patch.ndim > 2:
                patch = patch.squeeze()
            
            # 리사이즈 (원본 크기의 절반으로)
            # 간단한 다운샘플링 (평균 풀링 방식)
            step_y = ph // display_ph
            step_x = pw // display_pw
            patch_resized = patch[::step_y, ::step_x]
            
            # 크기가 정확히 맞지 않으면 자르기
            if patch_resized.shape[0] > display_ph:
                patch_resized = patch_resized[:display_ph, :]
            if patch_resized.shape[1] > display_pw:
                patch_resized = patch_resized[:, :display_pw]
            
            # 크기가 작으면 패딩
            if patch_resized.shape[0] < display_ph or patch_resized.shape[1] < display_pw:
                padded = np.zeros((display_ph, display_pw), dtype=patch_resized.dtype)
                padded[:patch_resized.shape[0], :patch_resized.shape[1]] = patch_resized
                patch_resized = padded
            
            # 패치 배치
            vis_img[y_start:y_end, x_start:x_end] = patch_resized
            
            patch_idx += 1
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # 왼쪽: 원본 이미지
    ax1 = axes[0]
    ax1.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax1.set_title(f'Original Image\nSize: {h} x {w} pixels', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # 패치 경계선 그리기 (원본 이미지에)
    if show_boundaries:
        for row in range((padded_h // ph) + 1):
            y = row * ph
            if y <= h:
                ax1.axhline(y=y, color='red', linewidth=1, alpha=0.5, linestyle='--')
        for col in range((padded_w // pw) + 1):
            x = col * pw
            if x <= w:
                ax1.axvline(x=x, color='red', linewidth=1, alpha=0.5, linestyle='--')
    
    # 오른쪽: 패치 그리드
    ax2 = axes[1]
    ax2.imshow(vis_img, cmap='gray', vmin=0, vmax=1)
    ax2.set_title(f'Patch Grid\nTotal: {n_patches} patches ({n_rows} rows x {n_cols} cols)\nPatch size: {ph} x {pw} pixels', 
                  fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # 패치 번호 표시 (작은 패치들에)
    patch_idx = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if patch_idx >= n_patches:
                break
            
            y_center = row * (display_ph + border_width) + border_width + display_ph // 2
            x_center = col * (display_pw + border_width) + border_width + display_pw // 2
            
            # 패치 번호 표시 (작은 텍스트)
            ax2.text(x_center, y_center, str(patch_idx), 
                    ha='center', va='center', 
                    fontsize=8, color='yellow', 
                    weight='bold',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            
            patch_idx += 1
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Patch visualization saved: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize patch splitting results of original image")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Input image path (.npy file or .tif file)")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output image path (default: input_filename_patches.png)")
    parser.add_argument("--patch_size", type=int, default=512,
                        help="Patch size (default: 512)")
    parser.add_argument("--max_patches_per_row", type=int, default=None,
                        help="Maximum number of patches per row (default: auto-calculate)")
    parser.add_argument("--no_boundaries", action="store_true",
                        help="Do not show patch boundaries")
    
    args = parser.parse_args()
    
    # 입력 파일 로드
    if args.input_path.endswith('.npy'):
        img = np.load(args.input_path)
    elif args.input_path.endswith('.tif'):
        import rasterio
        with rasterio.open(args.input_path) as src:
            img = src.read(1).astype(np.float32)
            # 정규화 (0~1 범위로)
            if img.max() > 1:
                img = img / img.max()
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {args.input_path}")
    
    print(f"Input image loaded: {img.shape}")
    
    # 패치 분할
    patch_size = (args.patch_size, args.patch_size)
    patches = split_into_patches_with_padding(img, patch_size)
    print(f"Patch splitting completed: {len(patches)} patches")
    
    # 출력 경로 결정
    if args.output_path is None:
        base_name = os.path.splitext(os.path.basename(args.input_path))[0]
        output_dir = os.path.dirname(args.input_path)
        args.output_path = os.path.join(output_dir, f"{base_name}_patches.png")
    
    # 시각화
    visualize_patch_grid(
        img, patches, patch_size,
        save_path=args.output_path,
        show_boundaries=not args.no_boundaries,
        max_patches_per_row=args.max_patches_per_row
    )
    
    print(f"\nCompleted! Output file: {args.output_path}")


if __name__ == "__main__":
    main()

