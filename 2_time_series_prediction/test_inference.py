"""
ConvLSTM 모델 추론 테스트 스크립트
- conv_lstm_infer.ipynb 노트북의 Python 파일 버전
- 학습된 모델로 validation 데이터에 대한 예측 수행 및 시각화
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

from data_preprocess import load_and_split_data, create_shifted_frames, compress_weather_data


# ===============================
# 설정
# ===============================

# 기본 경로 설정
IMG_SEQUENCES_PATH = "./data/data.npy"
WEATHER_DATA_PATH = "./data/weather_ex.csv"
LOAD_WEIGHT_DIR = "./weights"
LOAD_WEIGHT_PATH = "pred_with_weather.weights.h5"

# 기상 데이터 압축 설정
DURATIONS = [1, 2, 2, 1]


# ===============================
# 모델 빌드 함수
# ===============================

def build_model(input_img_shape: tuple, input_weather_dim: int, filters: int = 16):
    """
    ConvLSTM 인코더-디코더 모델 구축
    (노트북과 동일한 구조, filters=32)
    """
    # 1. 입력 레이어 정의
    image_input = layers.Input(shape=(None, *input_img_shape), name='image_input')
    weather_input = layers.Input(shape=(None, input_weather_dim), name='weather_input')
    
    # 2. 인코더: Conv-LSTM으로 이미지 다운샘플링 (512x512 -> 256x256 -> 128x128)
    x_img = layers.ConvLSTM2D(filters, (5, 5), padding="same", strides=(2, 2), 
                              return_sequences=True, activation="relu")(image_input)
    x_img = layers.LayerNormalization()(x_img)
    x_img = layers.ConvLSTM2D(filters, (3, 3), padding="same", strides=(2, 2), 
                              return_sequences=True, activation="relu")(x_img)
    x_img = layers.LayerNormalization()(x_img)
    
    # 3. 날씨 데이터 결합
    h, w = x_img.shape[2], x_img.shape[3]
    weather_tiled = layers.TimeDistributed(layers.Dense(h * w * 1))(weather_input)
    weather_reshaped = layers.Reshape((x_img.shape[1], h, w, 1))(weather_tiled)
    combined_features = layers.Concatenate(axis=-1)([x_img, weather_reshaped])
    
    # 4. 디코더: Conv3DTranspose로 이미지 업샘플링 (128x128 -> 512x512)
    x_decode = layers.Conv3DTranspose(filters, (3, 3, 3), padding="same", 
                                       strides=(1, 2, 2), activation="relu")(combined_features)
    x_decode = layers.LayerNormalization()(x_decode)
    x_decode = layers.Conv3DTranspose(filters, (3, 3, 3), padding="same", 
                                       strides=(1, 2, 2), activation="relu")(x_decode)
    x_decode = layers.LayerNormalization()(x_decode)
    
    # 최종 예측 (결과는 원본 이미지와 동일한 크기)
    out = layers.Conv3D(1, (3, 3, 3), padding="same", activation="sigmoid")(x_decode)
    
    # 모델 정의 및 컴파일
    model = keras.models.Model(inputs=[image_input, weather_input], outputs=out)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.binary_crossentropy,
        run_eagerly=True
    )
    
    return model


# ===============================
# 시각화 함수
# ===============================

def visualize_sequence(seq: np.ndarray, title: str = "Sequence", save_path: str = None):
    """
    이미지 시퀀스 시각화
    
    Parameters
    ----------
    seq : np.ndarray
        이미지 시퀀스 (time, height, width, channels)
    title : str
        그래프 제목
    save_path : str, optional
        저장 경로
    """
    seq_len = seq.shape[0]
    fig, axes = plt.subplots(1, seq_len, figsize=(3 * seq_len, 3))
    fig.suptitle(title, fontsize=14)
    
    if seq_len == 1:
        axes = [axes]
    
    for t in range(seq_len):
        ax = axes[t]
        frame = seq[t].squeeze()
        ax.imshow(frame, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        ax.set_title(f"Frame {t}")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_prediction_comparison(true_frames: np.ndarray, predicted_frames: np.ndarray,
                                     input_len: int, sample_idx: int, 
                                     save_path: str = None):
    """
    예측 결과와 Ground Truth 비교 시각화
    
    Parameters
    ----------
    true_frames : np.ndarray
        Ground truth 프레임들
    predicted_frames : np.ndarray
        예측된 프레임들
    input_len : int
        입력 프레임 수
    sample_idx : int
        샘플 인덱스
    save_path : str, optional
        저장 경로
    """
    num_to_predict = predicted_frames.shape[0]
    
    fig, axes = plt.subplots(2, num_to_predict, figsize=(3 * num_to_predict, 6))
    axes = np.atleast_2d(axes)
    
    # True frames (실제 정답)
    for idx in range(num_to_predict):
        ax = axes[0, idx]
        ax.imshow(true_frames[idx].squeeze(), cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"True F{idx + input_len + 1}")
        ax.axis('off')
    
    # Predicted frames (모델 예측)
    for idx in range(num_to_predict):
        ax = axes[1, idx]
        ax.imshow(predicted_frames[idx].squeeze(), cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"Pred F{idx + input_len + 1}")
        ax.axis('off')
    
    plt.suptitle(f"Sample {sample_idx}: True vs. Predicted ({input_len}-to-{num_to_predict})", 
                 fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_full_comparison(input_frames: np.ndarray, true_frames: np.ndarray, 
                               predicted_frames: np.ndarray, sample_idx: int,
                               save_path: str = None):
    """
    입력, Ground Truth, 예측 전체 비교 시각화
    
    Parameters
    ----------
    input_frames : np.ndarray
        입력 프레임들
    true_frames : np.ndarray
        Ground truth 프레임들
    predicted_frames : np.ndarray
        예측된 프레임들
    sample_idx : int
        샘플 인덱스
    save_path : str, optional
        저장 경로
    """
    n_input = input_frames.shape[0]
    n_pred = predicted_frames.shape[0]
    total_cols = max(n_input, n_pred)
    
    fig, axes = plt.subplots(3, total_cols, figsize=(3 * total_cols, 9))
    fig.suptitle(f"Sample {sample_idx}: Full Comparison", fontsize=14)
    
    # Row 1: Input frames
    for i in range(total_cols):
        ax = axes[0, i]
        if i < n_input:
            ax.imshow(input_frames[i].squeeze(), cmap='gray', vmin=0, vmax=1)
            ax.set_title(f"Input {i+1}")
        ax.axis('off')
    axes[0, 0].set_ylabel("Input", fontsize=12)
    
    # Row 2: Ground Truth
    for i in range(total_cols):
        ax = axes[1, i]
        if i < n_pred:
            ax.imshow(true_frames[i].squeeze(), cmap='gray', vmin=0, vmax=1)
            ax.set_title(f"GT {n_input + i + 1}")
        ax.axis('off')
    axes[1, 0].set_ylabel("Ground Truth", fontsize=12)
    
    # Row 3: Predictions
    for i in range(total_cols):
        ax = axes[2, i]
        if i < n_pred:
            ax.imshow(predicted_frames[i].squeeze(), cmap='gray', vmin=0, vmax=1)
            ax.set_title(f"Pred {n_input + i + 1}")
        ax.axis('off')
    axes[2, 0].set_ylabel("Prediction", fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ===============================
# 추론 함수
# ===============================

def autoregressive_predict(model, seq: np.ndarray, weather_seq: np.ndarray,
                           input_len: int, num_to_predict: int) -> np.ndarray:
    """
    자기회귀 방식으로 프레임 예측
    
    Parameters
    ----------
    model : keras.Model
        학습된 모델
    seq : np.ndarray
        입력 시퀀스 (time, height, width, channels)
    weather_seq : np.ndarray
        기상 데이터 (time, features)
    input_len : int
        입력 프레임 수
    num_to_predict : int
        예측할 프레임 수
    
    Returns
    -------
    np.ndarray : 예측된 프레임들
    """
    all_predicted_frames = np.copy(seq[:input_len, ...])
    
    for i in range(num_to_predict):
        # 1. 현재 입력 시퀀스 준비
        current_image_input = np.expand_dims(all_predicted_frames[-input_len:, ...], axis=0)
        current_weather_input = np.expand_dims(weather_seq[i:i + input_len, ...], axis=0)
        
        # 2. 모델 예측 실행
        new_prediction = model.predict({
            'image_input': current_image_input,
            'weather_input': current_weather_input
        }, verbose=0)
        
        # 3. 예측 결과에서 다음 프레임만 추출
        new_prediction = np.squeeze(new_prediction, axis=0)
        predicted_frame = new_prediction[-1, ...]
        
        # 4. 예측된 프레임을 전체 예측 시퀀스에 추가
        all_predicted_frames = np.concatenate(
            (all_predicted_frames, np.expand_dims(predicted_frame, axis=0)), axis=0
        )
    
    # 예측된 프레임들만 반환
    return all_predicted_frames[-num_to_predict:]


# ===============================
# 메인 실행
# ===============================

def parse_args():
    parser = argparse.ArgumentParser(description="ConvLSTM 모델 추론 테스트")
    parser.add_argument("--data_path", type=str, default=IMG_SEQUENCES_PATH,
                        help="시퀀스 데이터 경로")
    parser.add_argument("--weather_path", type=str, default=WEATHER_DATA_PATH,
                        help="기상 데이터 경로")
    parser.add_argument("--weight_dir", type=str, default=LOAD_WEIGHT_DIR,
                        help="모델 가중치 디렉토리")
    parser.add_argument("--weight_name", type=str, default=LOAD_WEIGHT_PATH,
                        help="모델 가중치 파일명")
    parser.add_argument("--output_dir", type=str, default="./test_results",
                        help="결과 저장 디렉토리")
    parser.add_argument("--filters", type=int, default=16,
                        help="ConvLSTM 필터 수 (학습 기본값: 16)")
    parser.add_argument("--input_len", type=int, default=3,
                        help="입력 프레임 수")
    parser.add_argument("--num_to_predict", type=int, default=2,
                        help="예측할 프레임 수")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="테스트할 샘플 수")
    parser.add_argument("--seed", type=int, default=None,
                        help="랜덤 시드 (None이면 랜덤)")
    parser.add_argument("--save_figures", action="store_true",
                        help="시각화 결과 저장 여부")
    parser.add_argument("--show_figures", action="store_true",
                        help="시각화 결과 화면 표시 여부")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "=" * 60)
    print(" ConvLSTM 모델 추론 테스트")
    print("=" * 60)
    
    # 랜덤 시드 설정
    if args.seed is not None:
        np.random.seed(args.seed)
    
    # 결과 디렉토리 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    if args.save_figures:
        os.makedirs(output_dir, exist_ok=True)
    
    # -------------------------------
    # 1. 데이터 준비
    # -------------------------------
    print("\n[1/4] 데이터 로드 중...")
    
    # 위성 이미지 시퀀스 데이터 로드, train/val 분할
    train_seqs, val_seqs = load_and_split_data(data_path=args.data_path)
    
    # sliding window 형식의 학습을 위해 x/y 분할
    x_train_seqs, y_train_seqs = create_shifted_frames(train_seqs)
    x_val_seqs, y_val_seqs = create_shifted_frames(val_seqs)
    
    # 기상 데이터 로드
    raw_weather_data = pd.read_csv(args.weather_path)
    weather_data = raw_weather_data.iloc[:6, [2, 3, 4, 5, 6, 7]]
    
    # 기상 데이터 정규화
    weather_min, weather_max = weather_data.min(), weather_data.max()
    weather_data_scaled = (weather_data - weather_min) / (weather_max - weather_min)
    
    # 6개 일자 데이터를 4개 간격 데이터로 재구성
    compressed_weather_data = compress_weather_data(weather_data_scaled, durations=DURATIONS)
    
    # 위성 이미지 시퀀스 데이터 수에 맞춰 기상 데이터 준비
    x_val_weather = np.tile(compressed_weather_data, (x_val_seqs.shape[0], 1, 1))
    
    # 데이터셋 정보 출력
    print(f"  x_train_seqs: {x_train_seqs.shape}, y_train_seqs: {y_train_seqs.shape}")
    print(f"  x_val_seqs  : {x_val_seqs.shape}, y_val_seqs  : {y_val_seqs.shape}")
    print(f"  x_val_weather: {x_val_weather.shape}")
    
    # -------------------------------
    # 2. 모델 구축 및 가중치 로드
    # -------------------------------
    print("\n[2/4] 모델 로드 중...")
    
    model = build_model(
        input_img_shape=x_train_seqs.shape[2:],
        input_weather_dim=x_val_weather.shape[-1],
        filters=args.filters
    )
    
    weight_path = os.path.join(args.weight_dir, args.weight_name)
    model.load_weights(weight_path)
    print(f"  가중치 로드 완료: {weight_path}")
    
    # 모델 요약 출력
    print("\n[모델 구조]")
    model.summary()
    
    # -------------------------------
    # 3. 유효한 샘플 선택
    # -------------------------------
    print("\n[3/4] 테스트 샘플 선택 중...")
    
    # 모든 값이 0이 아닌 유효한 시퀀스의 인덱스들을 찾음
    valid_indices = [idx for idx, seq in enumerate(val_seqs) if np.sum(seq) != 0]
    print(f"  유효한 시퀀스 수: {len(valid_indices)} / {len(val_seqs)}")
    
    # 유효한 인덱스 중에서 무작위로 선택
    num_to_select = min(args.num_samples, len(valid_indices))
    ex_indices = np.random.choice(valid_indices, size=num_to_select, replace=False)
    print(f"  선택된 샘플 인덱스: {ex_indices}")
    
    # -------------------------------
    # 4. 추론 및 시각화
    # -------------------------------
    print("\n[4/4] 추론 및 시각화...")
    
    for sample_idx in ex_indices:
        print(f"\n  샘플 {sample_idx} 처리 중...")
        
        seq = val_seqs[sample_idx]
        weather_seq = x_val_weather[sample_idx]
        
        # 입력 시퀀스 시각화
        if args.show_figures or args.save_figures:
            save_path = os.path.join(output_dir, f"sample_{sample_idx}_input.png") if args.save_figures else None
            visualize_sequence(seq, title=f"Sample {sample_idx}: Full Sequence", save_path=save_path)
            if args.show_figures and not args.save_figures:
                plt.show()
        
        # 자기회귀 예측
        true_frames = seq[args.input_len:args.input_len + args.num_to_predict, ...]
        predicted_frames = autoregressive_predict(
            model, seq, weather_seq,
            input_len=args.input_len,
            num_to_predict=args.num_to_predict
        )
        
        # 예측 결과 비교 시각화
        if args.show_figures or args.save_figures:
            save_path = os.path.join(output_dir, f"sample_{sample_idx}_comparison.png") if args.save_figures else None
            visualize_prediction_comparison(
                true_frames, predicted_frames,
                input_len=args.input_len,
                sample_idx=sample_idx,
                save_path=save_path
            )
            if args.show_figures and not args.save_figures:
                plt.show()
        
        # 전체 비교 시각화
        if args.show_figures or args.save_figures:
            save_path = os.path.join(output_dir, f"sample_{sample_idx}_full.png") if args.save_figures else None
            visualize_full_comparison(
                input_frames=seq[:args.input_len],
                true_frames=true_frames,
                predicted_frames=predicted_frames,
                sample_idx=sample_idx,
                save_path=save_path
            )
            if args.show_figures and not args.save_figures:
                plt.show()
    
    # 결과 저장 정보 출력
    if args.save_figures:
        print(f"\n결과 저장 완료: {output_dir}")
    
    print("\n" + "=" * 60)
    print(" 테스트 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()

