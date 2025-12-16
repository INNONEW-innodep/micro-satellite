"""
README에 사용할 예시 이미지 생성 스크립트
각 단계별 중간 결과물을 시각화하여 README에 추가할 이미지 생성
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# 한글 폰트 설정 (선택)
plt.rcParams['font.family'] = 'DejaVu Sans'


def create_example_images():
    """README 예시 이미지 생성"""
    
    # 출력 디렉토리
    output_dir = Path("docs/examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 데이터 준비 단계 예시
    print("1. 데이터 준비 단계 예시 생성...")
    create_data_preparation_examples(output_dir)
    
    # 2. 기상 데이터 예시
    print("2. 기상 데이터 예시 생성...")
    create_weather_data_example(output_dir)
    
    # 3. 패치 분할 예시
    print("3. 패치 분할 예시 생성...")
    create_patch_visualization_example(output_dir)
    
    # 4. 학습 곡선 예시
    print("4. 학습 곡선 예시 생성...")
    create_training_curve_example(output_dir)
    
    # 5. 평가 결과 예시
    print("5. 평가 결과 예시 생성...")
    create_evaluation_example(output_dir)
    
    # 6. 추론 결과 예시
    print("6. 추론 결과 예시 생성...")
    create_inference_example(output_dir)
    
    print(f"\n모든 예시 이미지 생성 완료: {output_dir}")


def create_data_preparation_examples(output_dir):
    """데이터 준비 단계 예시 이미지"""
    
    # 실제 변환된 이미지 사용
    img_dir = Path("data/busan/img")
    if img_dir.exists():
        img_files = sorted([f for f in img_dir.glob("*.png") if "patches" not in f.name])
        
        if len(img_files) >= 4:
            # 실제 이미지 사용
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            dates = ["2020-02-18", "2020-03-12", "2020-03-25", "2020-04-14"]
            
            for idx, (ax, date, img_file) in enumerate(zip(axes.flat, dates, img_files[:4])):
                img = plt.imread(img_file)
                ax.imshow(img, cmap='gray')
                ax.set_title(f'Time Point {idx+1}: {date}', fontsize=12)
                ax.axis('off')
            
            fig.suptitle('Converted Images Example (4 Time Points)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_dir / "02_converted_images.png", dpi=150, bbox_inches='tight')
            plt.close()
        else:
            # 더미 데이터 사용
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            dates = ["2020-02-18", "2020-03-12", "2020-03-25", "2020-04-14"]
            
            for idx, (ax, date) in enumerate(zip(axes.flat, dates)):
                img = np.random.rand(200, 200)
                img = (img > 0.85).astype(float)
                ax.imshow(img, cmap='gray', vmin=0, vmax=1)
                ax.set_title(f'Time Point {idx+1}: {date}', fontsize=12)
                ax.axis('off')
            
            fig.suptitle('Converted Images Example (4 Time Points)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_dir / "02_converted_images.png", dpi=150, bbox_inches='tight')
            plt.close()
    else:
        print(f"Warning: {img_dir} not found, skipping converted images example")


def create_weather_data_example(output_dir):
    """기상 데이터 예시"""
    
    # CSV 데이터 예시 생성
    data = {
        '일자(date)': ['2020. 2. 18', '2020. 2. 19', '2020. 2. 20', '2020. 2. 21', '2020. 2. 22'],
        '간격 (Δt)': ['', '', '', '', ''],
        'humidity (%)': [55, 58, 60, 62, 65],
        'precipitation (mm)': [0.3, 2.4, 0.0, 5.2, 1.1],
        'tmp_min (°C)': [17, 18, 19, 16, 15],
        'tmp_max (°C)': [27, 28, 29, 25, 24],
        'pressure (hPa)': [1020, 1020, 1018, 1015, 1016],
        'wind_speed (m/s)': [3, 3, 4, 5, 4]
    }
    
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns,
                     cellLoc='center', loc='center',
                     colWidths=[0.15, 0.1, 0.12, 0.15, 0.12, 0.12, 0.12, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # 헤더 스타일
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Weather Data CSV Format Example', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / "03_weather_data.png", dpi=150, bbox_inches='tight')
    plt.close()


def create_patch_visualization_example(output_dir):
    """패치 분할 예시"""
    
    # 실제 패치 시각화 이미지 사용
    patch_img_path = Path("data/busan/img/20200218_patches.png")
    if patch_img_path.exists():
        # 실제 이미지 복사
        import shutil
        shutil.copy(patch_img_path, output_dir / "04_patch_split.png")
        print(f"  Using actual patch visualization: {patch_img_path}")
        return
    
    # 더미 원본 이미지 (실제 이미지가 없을 경우)
    original = np.random.rand(512, 512)
    original = (original > 0.9).astype(float)
    
    # 패치로 분할 (4x4 그리드)
    patch_size = 128
    n_patches = 16
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # 왼쪽: 원본 이미지 + 패치 경계선
    ax1 = axes[0]
    ax1.imshow(original, cmap='gray', vmin=0, vmax=1)
    
    # 패치 경계선 그리기
    for i in range(5):
        ax1.axhline(y=i*patch_size, color='red', linewidth=2, linestyle='--', alpha=0.7)
        ax1.axvline(x=i*patch_size, color='red', linewidth=2, linestyle='--', alpha=0.7)
    
    ax1.set_title('Original Image (512×512)\nPatch Boundaries', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # 오른쪽: 패치 그리드
    ax2 = axes[1]
    grid_img = np.zeros((512, 512))
    
    patch_idx = 0
    for row in range(4):
        for col in range(4):
            y_start = row * patch_size
            y_end = y_start + patch_size
            x_start = col * patch_size
            x_end = x_start + patch_size
            
            patch = original[y_start:y_end, x_start:x_end]
            grid_img[y_start:y_end, x_start:x_end] = patch
            
            # 패치 번호 표시
            ax2.text(x_start + patch_size//2, y_start + patch_size//2, 
                    str(patch_idx), ha='center', va='center',
                    fontsize=10, color='yellow', weight='bold',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            
            patch_idx += 1
    
    ax2.imshow(grid_img, cmap='gray', vmin=0, vmax=1)
    ax2.set_title(f'Patch Splitting Result\nTotal: {n_patches} patches (4 rows × 4 cols)', 
                  fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    fig.suptitle('Patch Splitting Example', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "04_patch_split.png", dpi=150, bbox_inches='tight')
    plt.close()


def create_training_curve_example(output_dir):
    """학습 곡선 예시"""
    
    # 더미 학습 곡선 데이터
    epochs = np.arange(1, 31)
    train_loss = 0.5 * np.exp(-epochs/10) + 0.05 + np.random.normal(0, 0.01, len(epochs))
    val_loss = 0.6 * np.exp(-epochs/10) + 0.08 + np.random.normal(0, 0.015, len(epochs))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    ax.axvline(x=20, color='green', linestyle='--', alpha=0.7, label='Early Stopping')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Model Training Curve Example', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "05_training_curve.png", dpi=150, bbox_inches='tight')
    plt.close()


def create_evaluation_example(output_dir):
    """평가 결과 예시"""
    
    # 실제 평가 결과 이미지 사용
    eval_results_dir = Path("eval_results")
    if eval_results_dir.exists():
        # 가장 최근 평가 결과 찾기
        eval_dirs = sorted([d for d in eval_results_dir.iterdir() if d.is_dir()], reverse=True)
        if eval_dirs:
            metrics_img = eval_dirs[0] / "visualizations" / "metrics_summary.png"
            if metrics_img.exists():
                import shutil
                shutil.copy(metrics_img, output_dir / "06_evaluation_results.png")
                print(f"  Using actual evaluation results: {metrics_img}")
                return
    
    # 더미 평가 메트릭 (실제 이미지가 없을 경우)
    metrics = {
        'IoU': 0.7848,
        'Dice': 0.8160,
        'Pixel Accuracy': 0.9948,
        'Precision': 0.5612,
        'Recall': 0.4555,
        'F1 Score': 0.4827,
        'MSE': 0.0021,
        'MAE': 0.0456
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 왼쪽: 메트릭 바 차트
    names = list(metrics.keys())
    values = list(metrics.values())
    colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
    
    bars = ax1.barh(names, values, color=colors)
    ax1.set_xlabel('Score', fontsize=12)
    ax1.set_title('Evaluation Metrics Summary', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 1.1)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 값 표시
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax1.text(val + 0.02, i, f'{val:.4f}', va='center', fontsize=9)
    
    # 오른쪽: 예측 vs 실제 비교 (더미)
    ax2.axis('off')
    
    # 더미 이미지
    true_img = np.random.rand(200, 200)
    true_img = (true_img > 0.85).astype(float)
    pred_img = true_img.copy()
    pred_img[50:70, 50:70] = 1 - pred_img[50:70, 50:70]  # 일부 오차
    
    comparison = np.hstack([true_img, np.ones((200, 10)), pred_img])
    
    ax2.imshow(comparison, cmap='gray', vmin=0, vmax=1)
    ax2.set_title('Prediction vs Ground Truth Comparison', fontsize=12, fontweight='bold')
    ax2.text(100, 220, 'Ground Truth', ha='center', fontsize=10)
    ax2.text(310, 220, 'Prediction', ha='center', fontsize=10)
    
    fig.suptitle('Model Evaluation Results Example', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "06_evaluation_results.png", dpi=150, bbox_inches='tight')
    plt.close()


def create_inference_example(output_dir):
    """추론 결과 예시"""
    
    # 실제 추론 결과 이미지 사용
    inference_results_dir = Path("inference_results")
    if inference_results_dir.exists():
        # 가장 최근 추론 결과 찾기
        inference_dirs = sorted([d for d in inference_results_dir.iterdir() if d.is_dir()], reverse=True)
        if inference_dirs:
            # 개별 패치 시각화
            vis_dir = inference_dirs[0] / "visualizations"
            if vis_dir.exists():
                sample_files = sorted([f for f in vis_dir.glob("sample_*.png")])[:1]
                if sample_files:
                    import shutil
                    shutil.copy(sample_files[0], output_dir / "07_inference_result.png")
                    print(f"  Using actual inference visualization: {sample_files[0]}")
            
            # 전체 이미지 비교
            full_img_dir = inference_dirs[0] / "full_images"
            if full_img_dir.exists():
                full_comparison = full_img_dir / "full_comparison.png"
                if full_comparison.exists():
                    import shutil
                    shutil.copy(full_comparison, output_dir / "08_full_image_comparison.png")
                    print(f"  Using actual full image comparison: {full_comparison}")
                    return
    
    # 더미 데이터 (실제 이미지가 없을 경우)
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    
    # 입력 프레임 (3개)
    for i in range(3):
        img = np.random.rand(150, 150)
        img = (img > 0.85).astype(float)
        axes[0, i].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'Input Frame {i+1}', fontsize=10)
        axes[0, i].axis('off')
    
    # Ground Truth (1개)
    gt_img = np.random.rand(150, 150)
    gt_img = (gt_img > 0.85).astype(float)
    axes[1, 0].imshow(gt_img, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('Ground Truth', fontsize=10)
    axes[1, 0].axis('off')
    
    # 예측 결과 (1개)
    pred_img = gt_img.copy()
    pred_img[30:50, 30:50] = 1 - pred_img[30:50, 30:50]  # 일부 오차
    axes[1, 1].imshow(pred_img, cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title('Prediction', fontsize=10)
    axes[1, 1].axis('off')
    
    # 차이 (1개)
    diff = np.abs(gt_img - pred_img)
    axes[1, 2].imshow(diff, cmap='hot', vmin=0, vmax=1)
    axes[1, 2].set_title('Difference', fontsize=10)
    axes[1, 2].axis('off')
    
    # 나머지 빈 공간
    for i in range(3, 9):
        row = i // 3
        col = i % 3
        axes[row, col].axis('off')
    
    # 라벨
    axes[0, 0].set_ylabel('Input', fontsize=12, rotation=0, ha='right', va='center')
    axes[1, 0].set_ylabel('GT vs Pred', fontsize=12, rotation=0, ha='right', va='center')
    
    fig.suptitle('Inference Result Example: Input Frames 0,1,2 → Predict Frame 3', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "07_inference_result.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 전체 이미지 비교 예시
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    titles = ['Input Frame 0', 'Ground Truth (Frame 3)', 'Prediction (Frame 3)']
    for ax, title in zip(axes, titles):
        img = np.random.rand(300, 300)
        img = (img > 0.9).astype(float)
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
    
    fig.suptitle('Full Image Comparison Example (After Patch Merging)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "08_full_image_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    create_example_images()

