"""
데이터 정렬 상태 확인 스크립트
- data.npy의 각 프레임이 동일한 위치인지 시각적으로 확인
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


def verify_alignment(data_path: str, output_dir: str = None):
    """
    시퀀스 데이터의 정렬 상태 확인
    - 각 프레임의 차이(difference) 시각화
    - 정렬이 잘 되면 물 영역만 변화해야 함
    """
    
    data = np.load(data_path)
    print(f"데이터 shape: {data.shape}")
    print(f"  (n_patches, n_frames, height, width, channels)")
    
    n_patches = data.shape[0]
    n_frames = data.shape[1]
    
    # 비어있지 않은 패치 찾기
    non_empty_patches = []
    for i in range(n_patches):
        if np.sum(data[i]) > 0:
            non_empty_patches.append(i)
    
    print(f"\n비어있지 않은 패치 수: {len(non_empty_patches)} / {n_patches}")
    
    if len(non_empty_patches) == 0:
        print("⚠️ 모든 패치가 비어있습니다!")
        return
    
    # 샘플 패치 선택 (데이터가 있는 것 중 첫 번째)
    sample_idx = non_empty_patches[min(5, len(non_empty_patches)-1)]  # 5번째 또는 마지막
    sample_patch = data[sample_idx]
    
    print(f"\n샘플 패치 인덱스: {sample_idx}")
    print(f"샘플 패치 shape: {sample_patch.shape}")
    
    # 각 프레임 시각화
    fig, axes = plt.subplots(3, n_frames, figsize=(5*n_frames, 12))
    fig.suptitle(f'Alignment Verification (Patch #{sample_idx})', fontsize=16)
    
    # Row 0: 각 프레임 원본
    for i in range(n_frames):
        ax = axes[0, i]
        frame = sample_patch[i].squeeze()
        ax.imshow(frame, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'Frame {i+1}')
        ax.axis('off')
    axes[0, 0].set_ylabel('Original\nFrames', fontsize=12, rotation=0, ha='right', va='center')
    
    # Row 1: 첫 번째 프레임과의 차이
    ref_frame = sample_patch[0].squeeze()
    for i in range(n_frames):
        ax = axes[1, i]
        frame = sample_patch[i].squeeze()
        diff = np.abs(frame - ref_frame)
        ax.imshow(diff, cmap='hot', vmin=0, vmax=1)
        ax.set_title(f'|Frame{i+1} - Frame1|')
        ax.axis('off')
    axes[1, 0].set_ylabel('Diff from\nFrame 1', fontsize=12, rotation=0, ha='right', va='center')
    
    # Row 2: 연속 프레임 차이
    for i in range(n_frames):
        ax = axes[2, i]
        if i == 0:
            ax.text(0.5, 0.5, 'Reference', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
        else:
            prev_frame = sample_patch[i-1].squeeze()
            curr_frame = sample_patch[i].squeeze()
            diff = np.abs(curr_frame - prev_frame)
            ax.imshow(diff, cmap='hot', vmin=0, vmax=1)
            ax.set_title(f'|Frame{i+1} - Frame{i}|')
            ax.axis('off')
    axes[2, 0].set_ylabel('Diff from\nPrev Frame', fontsize=12, rotation=0, ha='right', va='center')
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, 'alignment_verification.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n저장됨: {save_path}")
        plt.close()
    else:
        plt.show()
    
    # 전체 패치의 프레임 간 차이 통계
    print("\n" + "=" * 60)
    print("프레임 간 차이 통계 (전체 패치)")
    print("=" * 60)
    
    for i in range(1, n_frames):
        diffs = []
        for patch_idx in non_empty_patches:
            prev = data[patch_idx, i-1].squeeze()
            curr = data[patch_idx, i].squeeze()
            diff = np.mean(np.abs(curr - prev))
            diffs.append(diff)
        
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)
        print(f"  Frame {i} → Frame {i+1}: 평균 차이 = {mean_diff:.4f} ± {std_diff:.4f}")
    
    # 정렬 상태 진단
    print("\n" + "=" * 60)
    print("정렬 상태 진단")
    print("=" * 60)
    
    # 첫 프레임과 마지막 프레임의 차이
    total_diff = []
    for patch_idx in non_empty_patches:
        first = data[patch_idx, 0].squeeze()
        last = data[patch_idx, -1].squeeze()
        diff = np.mean(np.abs(last - first))
        total_diff.append(diff)
    
    mean_total_diff = np.mean(total_diff)
    
    if mean_total_diff < 0.1:
        print(f"✓ 정렬 상태 양호: 프레임 간 평균 차이 = {mean_total_diff:.4f}")
        print("  물 영역의 시간적 변화만 감지됩니다.")
    elif mean_total_diff < 0.3:
        print(f"⚠️ 정렬 상태 의심: 프레임 간 평균 차이 = {mean_total_diff:.4f}")
        print("  약간의 위치 차이가 있을 수 있습니다.")
    else:
        print(f"❌ 정렬 불량: 프레임 간 평균 차이 = {mean_total_diff:.4f}")
        print("  이미지가 제대로 정렬되지 않았을 가능성이 높습니다.")
        print("  --align_geo 옵션을 사용하여 다시 변환해주세요.")


def compare_with_without_alignment(raster_dir: str, npy_dir: str, output_dir: str):
    """
    정렬 전/후 비교를 위한 시각화
    """
    import rasterio
    from rasterio.windows import from_bounds
    
    raster_files = sorted([f for f in os.listdir(raster_dir) if f.lower().endswith('.tif')])
    
    if len(raster_files) < 2:
        print("비교할 파일이 부족합니다.")
        return
    
    # 공통 영역 계산
    all_bounds = []
    for f in raster_files:
        with rasterio.open(os.path.join(raster_dir, f)) as src:
            all_bounds.append(src.bounds)
    
    common_left = max(b.left for b in all_bounds)
    common_right = min(b.right for b in all_bounds)
    common_bottom = max(b.bottom for b in all_bounds)
    common_top = min(b.top for b in all_bounds)
    
    print(f"공통 영역: ({common_left:.0f}, {common_bottom:.0f}) - ({common_right:.0f}, {common_top:.0f})")
    
    # 첫 두 이미지 비교
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Alignment Comparison (First Two Images)', fontsize=16)
    
    for i, f in enumerate(raster_files[:2]):
        raster_path = os.path.join(raster_dir, f)
        
        with rasterio.open(raster_path) as src:
            # 전체 이미지 (작게 샘플링)
            full_img = src.read(1, out_shape=(512, 512))
            
            # 공통 영역 클리핑
            window = from_bounds(common_left, common_bottom, common_right, common_top, src.transform)
            clipped_img = src.read(1, window=window)
            # 시각화용 리사이즈
            clipped_small = clipped_img[::max(1, clipped_img.shape[0]//512), 
                                        ::max(1, clipped_img.shape[1]//512)]
        
        # 전체 이미지
        axes[i, 0].imshow(full_img, cmap='gray')
        axes[i, 0].set_title(f'{f}\n(Full Image)')
        axes[i, 0].axis('off')
        
        # 클리핑된 이미지
        axes[i, 1].imshow(clipped_small, cmap='gray')
        axes[i, 1].set_title(f'{f}\n(Common Area Clipped)')
        axes[i, 1].axis('off')
    
    # 차이 이미지
    with rasterio.open(os.path.join(raster_dir, raster_files[0])) as src1:
        window1 = from_bounds(common_left, common_bottom, common_right, common_top, src1.transform)
        img1 = src1.read(1, window=window1)
    
    with rasterio.open(os.path.join(raster_dir, raster_files[1])) as src2:
        window2 = from_bounds(common_left, common_bottom, common_right, common_top, src2.transform)
        img2 = src2.read(1, window=window2)
    
    # 크기 맞추기
    min_h = min(img1.shape[0], img2.shape[0])
    min_w = min(img1.shape[1], img2.shape[1])
    img1 = img1[:min_h, :min_w]
    img2 = img2[:min_h, :min_w]
    
    diff = np.abs(img1.astype(float) - img2.astype(float))
    diff_small = diff[::max(1, diff.shape[0]//512), ::max(1, diff.shape[1]//512)]
    
    axes[0, 2].imshow(diff_small, cmap='hot')
    axes[0, 2].set_title('|Frame1 - Frame2| Difference')
    axes[0, 2].axis('off')
    
    axes[1, 2].text(0.5, 0.5, f'Mean Diff: {np.mean(diff):.4f}\nMax Diff: {np.max(diff):.4f}',
                    ha='center', va='center', transform=axes[1, 2].transAxes, fontsize=14)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, 'alignment_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n저장됨: {save_path}")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="데이터 정렬 상태 확인")
    parser.add_argument("--data_path", type=str, default="./data/busan/data.npy",
                        help="확인할 데이터 파일 경로")
    parser.add_argument("--output_dir", type=str, default="./data/busan",
                        help="결과 저장 디렉토리")
    parser.add_argument("--raster_dir", type=str, default=None,
                        help="원본 래스터 디렉토리 (비교 시각화용)")
    
    args = parser.parse_args()
    
    verify_alignment(args.data_path, args.output_dir)
    
    if args.raster_dir:
        compare_with_without_alignment(args.raster_dir, None, args.output_dir)

