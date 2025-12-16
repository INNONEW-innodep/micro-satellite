"""
데이터 값 범위 및 정렬 상태 상세 확인
"""
import numpy as np
import os


def check_data_values(data_path: str):
    """데이터 값 범위 확인"""
    
    data = np.load(data_path)
    print(f"데이터 shape: {data.shape}")
    print(f"데이터 타입: {data.dtype}")
    print(f"\n값 범위:")
    print(f"  최소값: {data.min()}")
    print(f"  최대값: {data.max()}")
    print(f"  평균값: {data.mean():.4f}")
    print(f"  표준편차: {data.std():.4f}")
    
    # 고유값 확인 (이진 마스크인지 확인)
    unique_vals = np.unique(data)
    print(f"\n고유값 개수: {len(unique_vals)}")
    if len(unique_vals) <= 10:
        print(f"고유값: {unique_vals}")
    
    # 이진 마스크인 경우
    if len(unique_vals) == 2:
        print("\n✓ 이진 마스크 데이터입니다 (0과 1)")
        
        # 1의 비율 (물 영역 비율)
        water_ratio = np.mean(data)
        print(f"  물 영역(1) 비율: {water_ratio:.4f} ({water_ratio*100:.2f}%)")
    
    # 프레임별 통계
    print("\n" + "=" * 60)
    print("프레임별 통계 (전체 패치 평균)")
    print("=" * 60)
    
    n_frames = data.shape[1]
    frame_means = []
    
    for i in range(n_frames):
        frame_data = data[:, i, :, :, :]
        mean_val = frame_data.mean()
        frame_means.append(mean_val)
        print(f"  Frame {i+1}: 평균 = {mean_val:.4f}, 최소 = {frame_data.min()}, 최대 = {frame_data.max()}")
    
    # 프레임 간 변화
    print("\n" + "=" * 60)
    print("프레임 간 변화 분석")
    print("=" * 60)
    
    for i in range(1, n_frames):
        prev = data[:, i-1, :, :, :]
        curr = data[:, i, :, :, :]
        
        # 차이 계산 (이진 마스크: 0과 1 사이 차이)
        diff = np.abs(curr.astype(float) - prev.astype(float))
        mean_diff = diff.mean()
        
        # 변화된 픽셀 비율 (이진 마스크에서 의미 있음)
        changed_pixels = np.sum(diff > 0)
        total_pixels = diff.size
        change_ratio = changed_pixels / total_pixels
        
        print(f"  Frame {i} → Frame {i+1}:")
        print(f"    평균 픽셀 차이: {mean_diff:.6f}")
        print(f"    변화된 픽셀 비율: {change_ratio:.4f} ({change_ratio*100:.2f}%)")
    
    # 정렬 상태 재평가
    print("\n" + "=" * 60)
    print("정렬 상태 재평가 (이진 마스크 기준)")
    print("=" * 60)
    
    first_frame = data[:, 0, :, :, :]
    last_frame = data[:, -1, :, :, :]
    
    total_diff = np.abs(last_frame.astype(float) - first_frame.astype(float))
    changed_ratio = np.sum(total_diff > 0) / total_diff.size
    
    print(f"첫 번째 ↔ 마지막 프레임 변화된 픽셀 비율: {changed_ratio:.4f} ({changed_ratio*100:.2f}%)")
    
    if changed_ratio < 0.1:
        print("\n✓ 정렬 상태 양호")
        print("  시간에 따른 물 영역 변화가 10% 미만입니다.")
        print("  이는 정상적인 시계열 변화 범위입니다.")
    elif changed_ratio < 0.3:
        print("\n⚠️ 중간 수준의 변화")
        print("  시간에 따른 물 영역 변화가 10~30%입니다.")
        print("  실제 수위 변화가 큰 경우일 수 있습니다.")
    else:
        print("\n⚠️ 변화가 큽니다")
        print("  시간에 따른 물 영역 변화가 30% 이상입니다.")
        print("  정렬 문제 또는 실제 큰 수위 변화일 수 있습니다.")
    
    # 비어있는 패치 분석
    print("\n" + "=" * 60)
    print("패치별 분석")
    print("=" * 60)
    
    n_patches = data.shape[0]
    empty_patches = 0
    partial_patches = 0
    full_patches = 0
    
    for i in range(n_patches):
        patch_sum = data[i].sum()
        if patch_sum == 0:
            empty_patches += 1
        elif patch_sum < data[i].size * 0.01:  # 1% 미만
            partial_patches += 1
        else:
            full_patches += 1
    
    print(f"  전체 패치 수: {n_patches}")
    print(f"  비어있는 패치 (물 없음): {empty_patches} ({empty_patches/n_patches*100:.1f}%)")
    print(f"  부분적 패치 (물 1% 미만): {partial_patches} ({partial_patches/n_patches*100:.1f}%)")
    print(f"  데이터 있는 패치: {full_patches} ({full_patches/n_patches*100:.1f}%)")


def check_npy_files(npy_dir: str):
    """개별 npy 파일 비교"""
    
    print("\n" + "=" * 60)
    print("개별 npy 파일 비교")
    print("=" * 60)
    
    npy_files = sorted([f for f in os.listdir(npy_dir) if f.endswith('.npy')])
    
    arrays = []
    for f in npy_files:
        arr = np.load(os.path.join(npy_dir, f))
        arrays.append(arr)
        print(f"\n{f}:")
        print(f"  Shape: {arr.shape}")
        print(f"  값 범위: {arr.min()} ~ {arr.max()}")
        print(f"  물 영역(1) 비율: {arr.mean():.4f}")
    
    # 프레임 간 비교
    if len(arrays) >= 2:
        print("\n" + "-" * 40)
        print("프레임 간 직접 비교:")
        
        for i in range(1, len(arrays)):
            prev = arrays[i-1].astype(float)
            curr = arrays[i].astype(float)
            
            # 크기가 같아야 비교 가능
            if prev.shape == curr.shape:
                diff = np.abs(curr - prev)
                changed = np.sum(diff > 0) / diff.size
                print(f"  {npy_files[i-1]} ↔ {npy_files[i]}: 변화 {changed*100:.2f}%")
            else:
                print(f"  ⚠️ 크기가 다름: {prev.shape} vs {curr.shape}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="데이터 값 확인")
    parser.add_argument("--data_path", type=str, default="./data/busan/data.npy",
                        help="확인할 데이터 파일 경로")
    parser.add_argument("--npy_dir", type=str, default="./data/busan/npy",
                        help="npy 파일 디렉토리")
    
    args = parser.parse_args()
    
    check_data_values(args.data_path)
    check_npy_files(args.npy_dir)

