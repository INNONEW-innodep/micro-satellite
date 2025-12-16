"""
ConvLSTM 모델 추론 스크립트
- 새로운 데이터에 대한 예측 수행
- 결과 시각화 및 저장
"""

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
tf.config.optimizer.set_jit(False)
import keras
from keras import layers

from data_preprocess import compress_weather_data, create_shifted_frames


# ===============================
# 모델 빌드 함수
# ===============================

def build_model(input_img_shape: tuple, input_weather_dim: int, filters: int = 16):
    """
    ConvLSTM 인코더-디코더 모델 구축
    """
    image_input = layers.Input(shape=(None, *input_img_shape), name='image_input')
    weather_input = layers.Input(shape=(None, input_weather_dim), name='weather_input')
    
    x_img = layers.ConvLSTM2D(filters, (5, 5), padding="same", strides=(2, 2), 
                              return_sequences=True, activation="relu")(image_input)
    x_img = layers.LayerNormalization()(x_img)
    x_img = layers.ConvLSTM2D(filters, (3, 3), padding="same", strides=(2, 2), 
                              return_sequences=True, activation="relu")(x_img)
    x_img = layers.LayerNormalization()(x_img)
    
    h, w = x_img.shape[2], x_img.shape[3]
    weather_tiled = layers.TimeDistributed(layers.Dense(h * w * 1))(weather_input)
    weather_reshaped = layers.Reshape((x_img.shape[1], h, w, 1))(weather_tiled)
    combined_features = layers.Concatenate(axis=-1)([x_img, weather_reshaped])
    
    x_decode = layers.Conv3DTranspose(filters, (3, 3, 3), padding="same", 
                                       strides=(1, 2, 2), activation="relu")(combined_features)
    x_decode = layers.LayerNormalization()(x_decode)
    x_decode = layers.Conv3DTranspose(filters, (3, 3, 3), padding="same", 
                                       strides=(1, 2, 2), activation="relu")(x_decode)
    x_decode = layers.LayerNormalization()(x_decode)
    
    out = layers.Conv3D(1, (3, 3, 3), padding="same", activation="sigmoid")(x_decode)
    
    model = keras.models.Model(inputs=[image_input, weather_input], outputs=out)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.binary_crossentropy,
    )
    
    return model


# ===============================
# 추론 함수
# ===============================

def predict_single_step(model, image_seq: np.ndarray, weather_seq: np.ndarray) -> np.ndarray:
    """
    단일 스텝 예측 (입력 시퀀스 -> 다음 시퀀스)
    
    Parameters
    ----------
    model : keras.Model
        학습된 모델
    image_seq : np.ndarray
        입력 이미지 시퀀스 (time, height, width, channels)
    weather_seq : np.ndarray
        입력 기상 데이터 (time, features)
    
    Returns
    -------
    np.ndarray : 예측된 이미지 시퀀스
    """
    image_input = np.expand_dims(image_seq, axis=0)
    weather_input = np.expand_dims(weather_seq, axis=0)
    
    prediction = model.predict({
        'image_input': image_input,
        'weather_input': weather_input
    }, verbose=0)
    
    return np.squeeze(prediction, axis=0)


def predict_autoregressive(model, initial_frames: np.ndarray, weather_data: np.ndarray,
                           num_to_predict: int, input_len: int = 3) -> np.ndarray:
    """
    자기회귀 방식 예측 (연속 프레임 생성)
    
    Parameters
    ----------
    model : keras.Model
        학습된 모델
    initial_frames : np.ndarray
        초기 입력 프레임들 (time, height, width, channels)
    weather_data : np.ndarray
        전체 기상 데이터 (time, features)
    num_to_predict : int
        예측할 프레임 수
    input_len : int
        모델 입력에 사용할 프레임 수
    
    Returns
    -------
    np.ndarray : 예측된 프레임들
    """
    all_frames = np.copy(initial_frames)
    
    for i in range(num_to_predict):
        # 현재 입력 준비
        current_image_input = all_frames[-input_len:]
        current_weather_input = weather_data[i:i + input_len]
        
        # 예측
        prediction = predict_single_step(model, current_image_input, current_weather_input)
        
        # 마지막 예측 프레임 추출 및 추가
        predicted_frame = prediction[-1:]
        all_frames = np.concatenate([all_frames, predicted_frame], axis=0)
    
    # 예측된 프레임들만 반환
    return all_frames[-num_to_predict:]


def predict_batch(model, image_batch: np.ndarray, weather_batch: np.ndarray) -> np.ndarray:
    """
    배치 단위 예측
    
    Parameters
    ----------
    model : keras.Model
        학습된 모델
    image_batch : np.ndarray
        입력 이미지 배치 (batch, time, height, width, channels)
    weather_batch : np.ndarray
        입력 기상 데이터 배치 (batch, time, features)
    
    Returns
    -------
    np.ndarray : 예측 결과 배치
    """
    return model.predict({
        'image_input': image_batch,
        'weather_input': weather_batch
    }, verbose=0)


# ===============================
# 시각화 함수
# ===============================

def visualize_prediction(input_frames: np.ndarray, predicted_frames: np.ndarray,
                         ground_truth: np.ndarray = None, save_path: str = None,
                         title: str = "Prediction Result", input_len: int = 3):
    """
    예측 결과 시각화 (입력, GT, 예측 비교)
    
    Parameters
    ----------
    input_frames : np.ndarray
        입력 프레임들
    predicted_frames : np.ndarray
        예측된 프레임들
    ground_truth : np.ndarray, optional
        Ground truth 프레임들 (있는 경우)
    save_path : str, optional
        저장 경로 (None이면 화면에 표시)
    title : str
        그래프 제목
    input_len : int
        입력 프레임 수 (프레임 번호 표시용)
    """
    n_input = input_frames.shape[0]
    n_pred = predicted_frames.shape[0]
    has_gt = ground_truth is not None
    
    n_rows = 3 if has_gt else 2
    n_cols = max(n_input, n_pred)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    fig.suptitle(title, fontsize=14)
    
    # axes가 1D인 경우 2D로 변환
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Row 0: 입력 프레임
    for i in range(n_cols):
        ax = axes[0, i]
        if i < n_input:
            ax.imshow(input_frames[i].squeeze(), cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'Input F{i+1}')
        ax.axis('off')
    axes[0, 0].set_ylabel("Input", fontsize=12, rotation=0, ha='right', va='center')
    
    # Row 1: Ground Truth (있는 경우) 또는 예측
    if has_gt:
        for i in range(n_cols):
            ax = axes[1, i]
            if i < ground_truth.shape[0]:
                ax.imshow(ground_truth[i].squeeze(), cmap='gray', vmin=0, vmax=1)
                ax.set_title(f'GT F{input_len + i + 1}')
            ax.axis('off')
        axes[1, 0].set_ylabel("Ground Truth", fontsize=12, rotation=0, ha='right', va='center')
        
        # Row 2: 예측
        for i in range(n_cols):
            ax = axes[2, i]
            if i < n_pred:
                ax.imshow(predicted_frames[i].squeeze(), cmap='gray', vmin=0, vmax=1)
                ax.set_title(f'Pred F{input_len + i + 1}')
            ax.axis('off')
        axes[2, 0].set_ylabel("Prediction", fontsize=12, rotation=0, ha='right', va='center')
    else:
        # GT 없으면 Row 1이 예측
        for i in range(n_cols):
            ax = axes[1, i]
            if i < n_pred:
                ax.imshow(predicted_frames[i].squeeze(), cmap='gray', vmin=0, vmax=1)
                ax.set_title(f'Pred F{input_len + i + 1}')
            ax.axis('off')
        axes[1, 0].set_ylabel("Prediction", fontsize=12, rotation=0, ha='right', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"시각화 저장됨: {save_path}")
    else:
        plt.show()


def visualize_comparison(predictions: list, ground_truths: list, indices: list,
                         save_dir: str = None):
    """
    여러 샘플의 예측 결과 비교 시각화
    
    Parameters
    ----------
    predictions : list
        예측 결과 리스트
    ground_truths : list
        Ground truth 리스트
    indices : list
        샘플 인덱스 리스트
    save_dir : str, optional
        저장 디렉토리
    """
    n_samples = len(predictions)
    
    for i, (pred, gt, idx) in enumerate(zip(predictions, ground_truths, indices)):
        n_frames = pred.shape[0]
        
        fig, axes = plt.subplots(2, n_frames, figsize=(3 * n_frames, 6))
        fig.suptitle(f'Sample {idx}', fontsize=14)
        
        for j in range(n_frames):
            # Ground truth
            axes[0, j].imshow(gt[j].squeeze(), cmap='gray', vmin=0, vmax=1)
            axes[0, j].set_title(f'GT Frame {j+1}')
            axes[0, j].axis('off')
            
            # Prediction
            axes[1, j].imshow(pred[j].squeeze(), cmap='gray', vmin=0, vmax=1)
            axes[1, j].set_title(f'Pred Frame {j+1}')
            axes[1, j].axis('off')
        
        plt.tight_layout()
        
        if save_dir:
            save_path = os.path.join(save_dir, f'comparison_sample_{idx}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def create_prediction_gif(frames: np.ndarray, save_path: str, fps: int = 2):
    """
    예측 프레임들을 GIF로 저장
    
    Parameters
    ----------
    frames : np.ndarray
        프레임 시퀀스 (time, height, width, channels)
    save_path : str
        저장 경로
    fps : int
        초당 프레임 수
    """
    try:
        import imageio
        
        # 0-255 범위로 변환
        frames_uint8 = (frames.squeeze() * 255).astype(np.uint8)
        
        imageio.mimsave(save_path, frames_uint8, fps=fps)
        print(f"GIF 저장됨: {save_path}")
    except ImportError:
        print("GIF 저장을 위해 imageio 패키지가 필요합니다: pip install imageio")


# ===============================
# 패치 병합 함수
# ===============================

def merge_patches_to_full_image(patches: np.ndarray, original_shape: tuple, 
                                  patch_size: tuple = (512, 512),
                                  grid_shape: tuple = None) -> np.ndarray:
    """
    패치들을 원본 크기의 전체 이미지로 병합
    
    Parameters
    ----------
    patches : np.ndarray
        패치 배열 (n_patches, height, width) 또는 (n_patches, height, width, channels)
    original_shape : tuple
        원본 이미지 크기 (height, width)
    patch_size : tuple
        패치 크기 (patch_height, patch_width)
    grid_shape : tuple, optional
        그리드 크기 (n_rows, n_cols). None이면 자동 계산
    
    Returns
    -------
    np.ndarray : 병합된 전체 이미지
    """
    ph, pw = patch_size
    orig_h, orig_w = original_shape
    
    # 그리드 크기 계산
    if grid_shape is not None:
        n_rows, n_cols = grid_shape
    else:
        # 패딩 고려하여 그리드 크기 계산
        n_rows = (orig_h + ph - 1) // ph  # ceil division
        n_cols = (orig_w + pw - 1) // pw
    
    # 패딩된 이미지 크기
    padded_h = n_rows * ph
    padded_w = n_cols * pw
    
    print(f"    Merge info: original={orig_h}x{orig_w}, grid={n_rows}x{n_cols}, patches={len(patches)}")
    
    # 전체 이미지는 항상 2D로 생성 (그레이스케일)
    full_image = np.zeros((padded_h, padded_w), dtype=np.float32)
    
    # 패치 병합 (행 우선 순서)
    patch_idx = 0
    for row in range(n_rows):
        for col in range(n_cols):
            y_start = row * ph
            y_end = y_start + ph
            x_start = col * pw
            x_end = x_start + pw
            
            if patch_idx < len(patches):
                # 패치를 2D로 squeeze
                patch = np.squeeze(patches[patch_idx])
                if patch.ndim > 2:
                    patch = patch[:, :, 0]  # 첫 번째 채널만 사용
                full_image[y_start:y_end, x_start:x_end] = patch
                patch_idx += 1
    
    # 원본 크기로 자르기 (패딩 제거)
    return full_image[:orig_h, :orig_w]


def merge_all_patches_for_frame(data: np.ndarray, frame_idx: int, 
                                 original_shape: tuple, patch_size: tuple = (512, 512)) -> np.ndarray:
    """
    특정 프레임의 모든 패치를 병합하여 전체 이미지 생성
    
    Parameters
    ----------
    data : np.ndarray
        전체 데이터 (n_patches, n_frames, height, width, channels)
    frame_idx : int
        프레임 인덱스
    original_shape : tuple
        원본 이미지 크기 (height, width)
    patch_size : tuple
        패치 크기
    
    Returns
    -------
    np.ndarray : 병합된 전체 이미지
    """
    # 해당 프레임의 모든 패치 추출
    patches = data[:, frame_idx, :, :, :]
    return merge_patches_to_full_image(patches, original_shape, patch_size)


def visualize_full_image_comparison(input_full: list, gt_full: list, pred_full: list,
                                      save_path: str = None, title: str = "Full Image Comparison"):
    """
    전체 이미지 비교 시각화 (입력, GT, 예측)
    
    Parameters
    ----------
    input_full : list
        입력 프레임들의 전체 이미지 리스트
    gt_full : list
        Ground Truth 프레임들의 전체 이미지 리스트
    pred_full : list
        예측 프레임들의 전체 이미지 리스트
    save_path : str, optional
        저장 경로
    title : str
        그래프 제목
    """
    n_input = len(input_full)
    n_pred = len(pred_full)
    n_cols = max(n_input, n_pred)
    
    has_gt = gt_full is not None and len(gt_full) > 0
    n_rows = 3 if has_gt else 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    fig.suptitle(title, fontsize=16)
    
    # axes 형태 보정
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Row 0: 입력 프레임
    for i in range(n_cols):
        ax = axes[0, i]
        if i < n_input:
            ax.imshow(input_full[i], cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'Input F{i+1}', fontsize=12)
        ax.axis('off')
    
    if has_gt:
        # Row 1: Ground Truth
        for i in range(n_cols):
            ax = axes[1, i]
            if i < len(gt_full):
                ax.imshow(gt_full[i], cmap='gray', vmin=0, vmax=1)
                ax.set_title(f'GT F{n_input + i + 1}', fontsize=12)
            ax.axis('off')
        
        # Row 2: 예측
        for i in range(n_cols):
            ax = axes[2, i]
            if i < n_pred:
                ax.imshow(pred_full[i], cmap='gray', vmin=0, vmax=1)
                ax.set_title(f'Pred F{n_input + i + 1}', fontsize=12)
            ax.axis('off')
    else:
        # Row 1: 예측
        for i in range(n_cols):
            ax = axes[1, i]
            if i < n_pred:
                ax.imshow(pred_full[i], cmap='gray', vmin=0, vmax=1)
                ax.set_title(f'Pred F{n_input + i + 1}', fontsize=12)
            ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"전체 이미지 시각화 저장됨: {save_path}")
    else:
        plt.show()


# ===============================
# 결과 저장 함수
# ===============================

def save_predictions(predictions: np.ndarray, save_path: str):
    """
    예측 결과를 NumPy 파일로 저장
    """
    np.save(save_path, predictions)
    print(f"예측 결과 저장됨: {save_path}")


def save_predictions_as_images(predictions: np.ndarray, save_dir: str, 
                                prefix: str = "pred", threshold: float = None):
    """
    예측 결과를 개별 이미지로 저장
    
    Parameters
    ----------
    predictions : np.ndarray
        예측 결과 (batch, time, height, width, channels) 또는 (time, height, width, channels)
    save_dir : str
        저장 디렉토리
    prefix : str
        파일명 접두사
    threshold : float, optional
        이진화 임계값 (None이면 원본 저장)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if predictions.ndim == 4:
        predictions = np.expand_dims(predictions, axis=0)
    
    for batch_idx in range(predictions.shape[0]):
        for frame_idx in range(predictions.shape[1]):
            frame = predictions[batch_idx, frame_idx].squeeze()
            
            if threshold is not None:
                frame = (frame > threshold).astype(np.float32)
            
            save_path = os.path.join(save_dir, f"{prefix}_b{batch_idx}_f{frame_idx}.png")
            plt.imsave(save_path, frame, cmap='gray', vmin=0, vmax=1)
    
    print(f"이미지 저장됨: {save_dir}")


# ===============================
# 메인 실행
# ===============================

def parse_args():
    parser = argparse.ArgumentParser(description="ConvLSTM 모델 추론")
    parser.add_argument("--data_path", type=str, required=True,
                        help="입력 시퀀스 데이터 경로 (.npy)")
    parser.add_argument("--weather_path", type=str, default="./data/weather_ex.csv",
                        help="기상 데이터 경로")
    parser.add_argument("--weight_path", type=str, default="./weights/pred_with_weather.weights.h5",
                        help="모델 가중치 경로")
    parser.add_argument("--output_dir", type=str, default="./inference_results",
                        help="결과 저장 디렉토리")
    parser.add_argument("--filters", type=int, default=16,
                        help="ConvLSTM 필터 수")
    parser.add_argument("--input_len", type=int, default=3,
                        help="입력 프레임 수")
    parser.add_argument("--num_to_predict", type=int, default=2,
                        help="예측할 프레임 수")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="이진화 임계값 (기본값: 0.5, 낮을수록 더 많은 픽셀을 물로 인식)")
    parser.add_argument("--sample_indices", type=str, default=None,
                        help="추론할 샘플 인덱스 (쉼표 구분, 예: 0,1,2)")
    parser.add_argument("--save_gif", action="store_true",
                        help="GIF 저장 여부")
    parser.add_argument("--no_visualize", action="store_true",
                        help="시각화 저장 비활성화")
    parser.add_argument("--merge_patches", action="store_true",
                        help="패치들을 전체 이미지로 병합하여 시각화")
    parser.add_argument("--original_height", type=int, default=None,
                        help="원본 이미지 높이 (패치 병합 시 필요)")
    parser.add_argument("--original_width", type=int, default=None,
                        help="원본 이미지 너비 (패치 병합 시 필요)")
    parser.add_argument("--patch_size", type=int, default=512,
                        help="패치 크기 (기본: 512)")
    parser.add_argument("--prediction_mode", type=str, default="autoregressive",
                        choices=["autoregressive", "standard"],
                        help="예측 모드: 'autoregressive' (자기회귀) 또는 'standard' (표준, evaluate.py와 동일)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "=" * 60)
    print(" ConvLSTM 모델 추론 시작")
    print("=" * 60)
    
    # 결과 디렉토리 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # -------------------------------
    # 1. 데이터 로드
    # -------------------------------
    print("\n[1/4] 데이터 로드 중...")
    
    data = np.load(args.data_path)
    print(f"  입력 데이터: {data.shape}")
    
    # 기상 데이터
    raw_weather_data = pd.read_csv(args.weather_path)
    weather_data = raw_weather_data.iloc[:6, [2, 3, 4, 5, 6, 7]]
    
    weather_min, weather_max = weather_data.min(), weather_data.max()
    weather_data_scaled = (weather_data - weather_min) / (weather_max - weather_min)
    
    DURATIONS = [1, 2, 2, 1]
    compressed_weather = compress_weather_data(weather_data_scaled, durations=DURATIONS)
    
    # -------------------------------
    # 2. 모델 로드
    # -------------------------------
    print("\n[2/4] 모델 로드 중...")
    
    # 입력 shape 결정
    if data.ndim == 5:  # (batch, time, h, w, c)
        input_img_shape = data.shape[2:]
    else:  # (time, h, w, c)
        input_img_shape = data.shape[1:]
    
    model = build_model(
        input_img_shape=input_img_shape,
        input_weather_dim=compressed_weather.shape[-1],
        filters=args.filters
    )
    
    model.load_weights(args.weight_path)
    print(f"  가중치 로드 완료: {args.weight_path}")
    
    # -------------------------------
    # 3. 추론 수행
    # -------------------------------
    print("\n[3/4] 추론 수행 중...")
    
    # 샘플 인덱스 결정
    if args.sample_indices:
        sample_indices = [int(x) for x in args.sample_indices.split(',')]
    else:
        # 유효한 샘플 자동 선택 (빈 데이터 제외)
        if data.ndim == 5:
            valid_indices = [i for i in range(data.shape[0]) if np.sum(data[i]) > 0]
        else:
            valid_indices = [0] if np.sum(data) > 0 else []
        
        if len(valid_indices) == 0:
            print("  경고: 유효한 샘플이 없습니다!")
            # 유효한 샘플이 없어도 최대 20개 선택
            max_samples = min(20, data.shape[0] if data.ndim == 5 else 1)
            sample_indices = list(range(max_samples))
        else:
            # 더 많은 샘플 선택 (최대 30개)
            max_samples = min(30, len(valid_indices))
            sample_indices = valid_indices[:max_samples]
            print(f"  유효한 샘플 {len(valid_indices)}개 중 {len(sample_indices)}개 선택: {sample_indices}")
    
    all_predictions = []
    all_inputs = []
    all_ground_truths = []
    
    for idx in sample_indices:
        print(f"  샘플 {idx} 처리 중...")
        
        if data.ndim == 5:
            seq = data[idx]
        else:
            seq = data
        
        # 입력 프레임 추출
        initial_frames = seq[:args.input_len]
        
        # Ground Truth 프레임 추출 (입력 이후 프레임들)
        gt_end = args.input_len + args.num_to_predict
        if seq.shape[0] >= gt_end:
            ground_truth_frames = seq[args.input_len:gt_end]
        else:
            ground_truth_frames = None
        
        # 기상 데이터 준비
        weather_seq = compressed_weather[:args.input_len + args.num_to_predict]
        
        # 예측 모드에 따라 다르게 처리
        if args.prediction_mode == "standard":
            # 표준 모드: evaluate.py와 동일한 방식
            # create_shifted_frames와 동일하게 시점 0,1,2 → 시점 1,2,3 예측
            input_frames = seq[:args.input_len]  # 시점 0, 1, 2
            
            # 기상 데이터: evaluate.py와 동일하게 처리
            # compressed_weather[:-1]을 사용 (마지막 시점 제외, x에 맞춤)
            weather_input = compressed_weather[:args.input_len]
            
            # 모델 예측 (시점 0,1,2 입력 → 시점 1,2,3 예측)
            image_input = np.expand_dims(input_frames, axis=0)
            weather_input_expanded = np.expand_dims(weather_input, axis=0)
            
            prediction = model.predict({
                'image_input': image_input,
                'weather_input': weather_input_expanded
            }, verbose=0)
            
            # 예측 결과에서 필요한 프레임만 추출
            # prediction shape: (1, 3, 512, 512, 1) - 시점 1,2,3 예측
            # num_to_predict=1이면 시점 3만, num_to_predict=2이면 시점 2,3 사용
            predicted_frames = np.squeeze(prediction, axis=0)
            # 시점 1,2,3 중에서 마지막 num_to_predict개만 사용
            predicted_frames = predicted_frames[-args.num_to_predict:]
            
            # Ground Truth: 시점 1,2,3 중에서 마지막 num_to_predict개
            target_start = args.input_len  # 시점 3의 인덱스
            target_end = min(target_start + args.num_to_predict, seq.shape[0])
            if seq.shape[0] >= target_end:
                # 시점 1,2,3 중에서 마지막 num_to_predict개
                ground_truth_frames = seq[target_start:target_end]
            else:
                ground_truth_frames = None
                
        else:  # autoregressive
            # 자기회귀 예측
            predicted_frames = predict_autoregressive(
                model, initial_frames, weather_seq,
                num_to_predict=args.num_to_predict,
                input_len=args.input_len
            )
            # Ground Truth는 이미 위에서 설정됨
        
        all_predictions.append(predicted_frames)
        all_inputs.append(initial_frames)
        if args.prediction_mode == "standard":
            all_ground_truths.append(ground_truth_frames)
        else:
            all_ground_truths.append(ground_truth_frames)
    
    # -------------------------------
    # 4. 결과 저장
    # -------------------------------
    print("\n[4/4] 결과 저장 중...")
    
    # NumPy 파일로 저장
    predictions_array = np.array(all_predictions)
    save_predictions(predictions_array, os.path.join(output_dir, "predictions.npy"))
    
    # 이미지로 저장
    save_predictions_as_images(
        predictions_array, 
        os.path.join(output_dir, "images"),
        threshold=args.threshold
    )
    
    # 시각화 저장 (기본 활성화 - 원본, GT, 예측 비교)
    if not args.no_visualize:
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        for i, (inp, pred, gt, idx) in enumerate(zip(all_inputs, all_predictions, all_ground_truths, sample_indices)):
            save_path = os.path.join(vis_dir, f"sample_{idx}.png")
            visualize_prediction(inp, pred, ground_truth=gt, save_path=save_path,
                                title=f"Sample {idx}: Input → GT vs Prediction",
                                input_len=args.input_len)
    
    # GIF 저장
    if args.save_gif:
        for i, (inp, pred, idx) in enumerate(zip(all_inputs, all_predictions, sample_indices)):
            # 입력 + 예측 프레임 합치기
            full_seq = np.concatenate([inp, pred], axis=0)
            gif_path = os.path.join(output_dir, f"sample_{idx}.gif")
            create_prediction_gif(full_seq, gif_path)
    
    # 패치 병합하여 전체 이미지 시각화
    if args.merge_patches:
        print("\n[추가] 패치 병합하여 전체 이미지 생성 중...")
        print("  전체 패치를 예측하여 full_images 생성...")
        
        # 원본 크기 결정
        if args.original_height is None or args.original_width is None:
            # 패치 수로부터 원본 크기 추정
            n_patches = data.shape[0]
            patch_size = args.patch_size
            
            # 정사각형으로 가정하여 추정
            n_per_side = int(np.ceil(np.sqrt(n_patches)))
            estimated_size = n_per_side * patch_size
            orig_h = args.original_height if args.original_height else estimated_size
            orig_w = args.original_width if args.original_width else estimated_size
            print(f"  원본 크기 추정: {orig_h} x {orig_w} (패치 수: {n_patches})")
        else:
            orig_h = args.original_height
            orig_w = args.original_width
            print(f"  원본 크기: {orig_h} x {orig_w}")
        
        full_img_dir = os.path.join(output_dir, "full_images")
        os.makedirs(full_img_dir, exist_ok=True)
        
        n_frames = data.shape[1]  # 시퀀스 길이
        n_all_patches = data.shape[0]  # 전체 패치 수
        patch_sz = (args.patch_size, args.patch_size)
        
        # 각 프레임별로 전체 이미지 병합
        for frame_idx in range(n_frames):
            # 원본 데이터에서 전체 이미지 생성
            full_original = merge_all_patches_for_frame(data, frame_idx, (orig_h, orig_w), patch_sz)
            
            # 저장
            save_path = os.path.join(full_img_dir, f"original_frame_{frame_idx}.png")
            plt.imsave(save_path, full_original.squeeze(), cmap='gray', vmin=0, vmax=1)
        
        print(f"  원본 전체 이미지 {n_frames}개 저장됨")
        
        # 전체 패치에 대한 예측 수행
        print(f"  전체 {n_all_patches}개 패치 예측 중...")
        all_patch_predictions = []  # (n_patches, num_to_predict, h, w, c)
        all_patch_ground_truths = []  # (n_patches, num_to_predict, h, w, c)
        
        batch_size = 8  # 배치 크기
        for batch_start in range(0, n_all_patches, batch_size):
            batch_end = min(batch_start + batch_size, n_all_patches)
            batch_indices = list(range(batch_start, batch_end))
            print(f"    패치 {batch_start}~{batch_end-1} 예측 중... ({batch_end}/{n_all_patches})")
            
            batch_predictions = []
            batch_ground_truths = []
            
            for patch_idx in batch_indices:
                if data.ndim == 5:
                    seq = data[patch_idx]
                else:
                    seq = data
                
                # 입력 프레임 추출
                initial_frames = seq[:args.input_len]
                
                # Ground Truth 프레임 추출
                gt_end = args.input_len + args.num_to_predict
                if seq.shape[0] >= gt_end:
                    ground_truth_frames = seq[args.input_len:gt_end]
                else:
                    ground_truth_frames = None
                
                # 기상 데이터 준비
                weather_seq = compressed_weather[:args.input_len + args.num_to_predict]
                
                # 예측 모드에 따라 다르게 처리
                if args.prediction_mode == "standard":
                    input_frames = seq[:args.input_len]
                    weather_input = compressed_weather[:args.input_len]
                    
                    image_input = np.expand_dims(input_frames, axis=0)
                    weather_input_expanded = np.expand_dims(weather_input, axis=0)
                    
                    prediction = model.predict({
                        'image_input': image_input,
                        'weather_input': weather_input_expanded
                    }, verbose=0)
                    
                    predicted_frames = np.squeeze(prediction, axis=0)
                    predicted_frames = predicted_frames[-args.num_to_predict:]
                    
                    target_start = args.input_len
                    target_end = min(target_start + args.num_to_predict, seq.shape[0])
                    if seq.shape[0] >= target_end:
                        ground_truth_frames = seq[target_start:target_end]
                    else:
                        ground_truth_frames = None
                else:  # autoregressive
                    predicted_frames = predict_autoregressive(
                        model, initial_frames, weather_seq,
                        num_to_predict=args.num_to_predict,
                        input_len=args.input_len
                    )
                
                batch_predictions.append(predicted_frames)
                if ground_truth_frames is not None:
                    batch_ground_truths.append(ground_truth_frames)
                else:
                    batch_ground_truths.append(np.zeros((args.num_to_predict, args.patch_size, args.patch_size, 1)))
            
            all_patch_predictions.extend(batch_predictions)
            all_patch_ground_truths.extend(batch_ground_truths)
        
        all_patch_predictions = np.array(all_patch_predictions)
        all_patch_ground_truths = np.array(all_patch_ground_truths)
        print(f"  전체 패치 예측 완료: {all_patch_predictions.shape}")
        
        # 예측 결과 병합 (모든 패치의 예측을 병합)
        # 물 영역이 있는 프레임만 필터링
        threshold_for_filter = args.threshold if args.threshold is not None else 0.5
        valid_pred_frame_indices = []  # 물 영역이 있는 프레임 인덱스 저장
        
        # 예측 프레임별로 전체 이미지 생성
        for pred_frame_idx in range(args.num_to_predict):
            # 모든 패치의 예측을 수집
            pred_patches = all_patch_predictions[:, pred_frame_idx, :, :, :]
            gt_patches = all_patch_ground_truths[:, pred_frame_idx, :, :, :]
            
            # 전체 이미지로 병합
            full_pred = merge_patches_to_full_image(pred_patches, (orig_h, orig_w), patch_sz)
            full_gt = merge_patches_to_full_image(gt_patches, (orig_h, orig_w), patch_sz)
            
            # 물 영역이 있는지 체크 (threshold 이상인 픽셀이 있는지)
            has_water = np.any(full_pred >= threshold_for_filter)
            
            if has_water:
                # 물 영역이 있는 프레임만 저장
                save_path = os.path.join(full_img_dir, f"prediction_frame_{args.input_len + pred_frame_idx}.png")
                plt.imsave(save_path, full_pred.squeeze(), cmap='gray', vmin=0, vmax=1)
                valid_pred_frame_indices.append(pred_frame_idx)
                print(f"  예측 프레임 {args.input_len + pred_frame_idx} 저장됨 (물 영역 있음)")
            else:
                print(f"  예측 프레임 {args.input_len + pred_frame_idx} 스킵됨 (물 영역 없음)")
        
        print(f"  예측 전체 이미지 {len(valid_pred_frame_indices)}개 저장됨 (총 {args.num_to_predict}개 중)")
        
        # 비교 시각화 생성
        print("  전체 이미지 비교 시각화 생성 중...")
        
        # 입력 프레임들의 전체 이미지
        input_full_images = []
        for frame_idx in range(args.input_len):
            full_img = merge_all_patches_for_frame(data, frame_idx, (orig_h, orig_w), patch_sz)
            input_full_images.append(full_img.squeeze())
        
        # GT 프레임들의 전체 이미지 (물 영역이 있는 예측 프레임에 대응하는 것만)
        gt_full_images = []
        if len(valid_pred_frame_indices) > 0:
            for pred_frame_idx in valid_pred_frame_indices:
                # 전체 패치의 GT를 병합
                gt_patches = all_patch_ground_truths[:, pred_frame_idx, :, :, :]
                full_gt = merge_patches_to_full_image(gt_patches, (orig_h, orig_w), patch_sz)
                gt_full_images.append(full_gt.squeeze())
        
        # 예측 프레임들의 전체 이미지 (물 영역이 있는 프레임만 로드)
        pred_full_images = []
        for pred_frame_idx in valid_pred_frame_indices:
            pred_path = os.path.join(full_img_dir, f"prediction_frame_{args.input_len + pred_frame_idx}.png")
            if os.path.exists(pred_path):
                pred_img = plt.imread(pred_path)
                pred_full_images.append(pred_img)
        
        # 비교 시각화 저장
        comparison_path = os.path.join(full_img_dir, "full_comparison.png")
        visualize_full_image_comparison(
            input_full_images, gt_full_images, pred_full_images,
            save_path=comparison_path,
            title="Full Image: Input → GT vs Prediction"
        )
        
        print(f"  전체 이미지 저장 완료: {full_img_dir}")
    
    print(f"\n결과 저장 완료: {output_dir}")
    print("\n추론 완료!")


if __name__ == "__main__":
    main()

