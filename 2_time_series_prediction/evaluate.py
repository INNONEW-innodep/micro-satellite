"""
ConvLSTM 모델 평가 스크립트
- Validation 데이터셋에 대한 정량적 평가 수행
- 메트릭: IoU, Dice Score, Pixel Accuracy, Precision, Recall, F1
"""

import argparse
import os
import json
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
# 평가 메트릭 함수
# ===============================

def calculate_iou(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """
    IoU (Intersection over Union) 계산
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth 이미지 (0~1 범위)
    y_pred : np.ndarray
        예측 이미지 (0~1 범위)
    threshold : float
        이진화 임계값
    
    Returns
    -------
    float : IoU 점수
    """
    y_true_bin = (y_true > threshold).astype(np.float32)
    y_pred_bin = (y_pred > threshold).astype(np.float32)
    
    intersection = np.sum(y_true_bin * y_pred_bin)
    union = np.sum(y_true_bin) + np.sum(y_pred_bin) - intersection
    
    if union == 0:
        return 1.0 if np.sum(y_true_bin) == 0 else 0.0
    
    return intersection / union


def calculate_dice(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """
    Dice Score (F1 for segmentation) 계산
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth 이미지
    y_pred : np.ndarray
        예측 이미지
    threshold : float
        이진화 임계값
    
    Returns
    -------
    float : Dice 점수
    """
    y_true_bin = (y_true > threshold).astype(np.float32)
    y_pred_bin = (y_pred > threshold).astype(np.float32)
    
    intersection = np.sum(y_true_bin * y_pred_bin)
    total = np.sum(y_true_bin) + np.sum(y_pred_bin)
    
    if total == 0:
        return 1.0 if np.sum(y_true_bin) == 0 else 0.0
    
    return 2.0 * intersection / total


def calculate_pixel_accuracy(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """
    Pixel Accuracy 계산
    """
    y_true_bin = (y_true > threshold).astype(np.float32)
    y_pred_bin = (y_pred > threshold).astype(np.float32)
    
    correct = np.sum(y_true_bin == y_pred_bin)
    total = y_true_bin.size
    
    return correct / total


def calculate_precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> tuple:
    """
    Precision, Recall, F1 Score 계산
    
    Returns
    -------
    tuple : (precision, recall, f1)
    """
    y_true_bin = (y_true > threshold).astype(np.float32)
    y_pred_bin = (y_pred > threshold).astype(np.float32)
    
    tp = np.sum(y_true_bin * y_pred_bin)
    fp = np.sum((1 - y_true_bin) * y_pred_bin)
    fn = np.sum(y_true_bin * (1 - y_pred_bin))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


def calculate_mse_mae(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """
    MSE, MAE 계산 (연속값 비교용)
    
    Returns
    -------
    tuple : (mse, mae)
    """
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    return mse, mae


# ===============================
# 모델 빌드 함수
# ===============================

def build_model(input_img_shape: tuple, input_weather_dim: int, filters: int = 16, seq_len: int = None):
    """
    ConvLSTM 인코더-디코더 모델 구축
    
    Parameters
    ----------
    input_img_shape : tuple
        이미지 입력 shape (height, width, channels)
    input_weather_dim : int
        기상 데이터 feature 수
    filters : int
        ConvLSTM 필터 수
    seq_len : int, optional
        시퀀스 길이 (None이면 동적)
    
    Returns
    -------
    keras.Model
    """
    # 입력 레이어 (seq_len이 지정되면 고정, 아니면 동적)
    image_input = layers.Input(shape=(seq_len, *input_img_shape), name='image_input')
    weather_input = layers.Input(shape=(seq_len, input_weather_dim), name='weather_input')
    
    # 인코더: ConvLSTM으로 다운샘플링
    x_img = layers.ConvLSTM2D(filters, (5, 5), padding="same", strides=(2, 2), 
                              return_sequences=True, activation="relu")(image_input)
    x_img = layers.LayerNormalization()(x_img)
    x_img = layers.ConvLSTM2D(filters, (3, 3), padding="same", strides=(2, 2), 
                              return_sequences=True, activation="relu")(x_img)
    x_img = layers.LayerNormalization()(x_img)
    
    # 날씨 데이터 결합
    h, w = x_img.shape[2], x_img.shape[3]
    weather_tiled = layers.TimeDistributed(layers.Dense(h * w * 1))(weather_input)
    weather_reshaped = layers.Reshape((-1, h, w, 1))(weather_tiled)  # -1로 동적 처리
    combined_features = layers.Concatenate(axis=-1)([x_img, weather_reshaped])
    
    # 디코더: Conv3DTranspose로 업샘플링
    x_decode = layers.Conv3DTranspose(filters, (3, 3, 3), padding="same", 
                                       strides=(1, 2, 2), activation="relu")(combined_features)
    x_decode = layers.LayerNormalization()(x_decode)
    x_decode = layers.Conv3DTranspose(filters, (3, 3, 3), padding="same", 
                                       strides=(1, 2, 2), activation="relu")(x_decode)
    x_decode = layers.LayerNormalization()(x_decode)
    
    # 최종 출력
    out = layers.Conv3D(1, (3, 3, 3), padding="same", activation="sigmoid")(x_decode)
    
    model = keras.models.Model(inputs=[image_input, weather_input], outputs=out)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.binary_crossentropy,
    )
    
    return model


# ===============================
# 평가 함수
# ===============================

def evaluate_model(model, x_img: np.ndarray, x_weather: np.ndarray, y_true: np.ndarray,
                   threshold: float = 0.5, batch_size: int = 4) -> dict:
    """
    전체 데이터셋에 대한 모델 평가
    
    Parameters
    ----------
    model : keras.Model
        평가할 모델
    x_img : np.ndarray
        입력 이미지 시퀀스
    x_weather : np.ndarray
        입력 기상 데이터
    y_true : np.ndarray
        Ground truth 이미지 시퀀스
    threshold : float
        이진화 임계값
    batch_size : int
        배치 크기
    
    Returns
    -------
    dict : 평가 결과 (메트릭별 평균, 표준편차, 샘플별 결과)
    """
    n_samples = x_img.shape[0]
    
    # 샘플별 메트릭 저장
    metrics_per_sample = {
        'iou': [],
        'dice': [],
        'pixel_accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'mse': [],
        'mae': []
    }
    
    print(f"\n평가 시작: {n_samples}개 샘플")
    print("-" * 50)
    
    valid_count = 0  # 유효 샘플 카운터
    
    # 배치 단위로 예측
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_x_img = x_img[start_idx:end_idx]
        batch_x_weather = x_weather[start_idx:end_idx]
        batch_y_true = y_true[start_idx:end_idx]
        
        # 예측
        batch_y_pred = model.predict(
            {'image_input': batch_x_img, 'weather_input': batch_x_weather},
            verbose=0
        )
        
        # 샘플별 메트릭 계산
        for i in range(batch_y_pred.shape[0]):
            y_t = batch_y_true[i]
            y_p = batch_y_pred[i]
            
            # 빈 샘플 체크 (모든 프레임이 0인 경우)
            # y_t shape: (time, height, width, channels)
            if np.sum(y_t) == 0:
                # 빈 샘플은 스킵 (자기회귀 평가와 동일하게 처리)
                continue
            
            valid_count += 1
            metrics_per_sample['iou'].append(calculate_iou(y_t, y_p, threshold))
            metrics_per_sample['dice'].append(calculate_dice(y_t, y_p, threshold))
            metrics_per_sample['pixel_accuracy'].append(calculate_pixel_accuracy(y_t, y_p, threshold))
            
            prec, rec, f1 = calculate_precision_recall_f1(y_t, y_p, threshold)
            metrics_per_sample['precision'].append(prec)
            metrics_per_sample['recall'].append(rec)
            metrics_per_sample['f1'].append(f1)
            
            mse, mae = calculate_mse_mae(y_t, y_p)
            metrics_per_sample['mse'].append(mse)
            metrics_per_sample['mae'].append(mae)
        
        print(f"  진행: {end_idx}/{n_samples} 샘플 완료")
    
    # 통계 계산
    results = {
        'n_samples': n_samples,
        'n_valid_samples': valid_count,
        'threshold': threshold,
        'metrics': {}
    }
    
    for metric_name, values in metrics_per_sample.items():
        results['metrics'][metric_name] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'values': [float(v) for v in values]
        }
    
    return results


def evaluate_autoregressive(model, val_seqs: np.ndarray, weather_data: np.ndarray,
                            input_len: int = 3, num_to_predict: int = 2,
                            threshold: float = 0.5) -> dict:
    """
    자기회귀 방식 평가 (실제 추론 시나리오)
    
    Parameters
    ----------
    model : keras.Model
        평가할 모델
    val_seqs : np.ndarray
        Validation 시퀀스 데이터
    weather_data : np.ndarray
        기상 데이터
    input_len : int
        입력 프레임 수
    num_to_predict : int
        예측할 프레임 수
    threshold : float
        이진화 임계값
    
    Returns
    -------
    dict : 자기회귀 평가 결과
    """
    n_samples = val_seqs.shape[0]
    
    metrics_per_sample = {
        'iou': [],
        'dice': [],
        'pixel_accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    # 프레임별 메트릭
    metrics_per_frame = {f'frame_{i+1}': {'iou': [], 'dice': []} 
                         for i in range(num_to_predict)}
    
    print(f"\n자기회귀 평가 시작: {n_samples}개 시퀀스")
    print(f"  입력 프레임: {input_len}, 예측 프레임: {num_to_predict}")
    print("-" * 50)
    
    valid_count = 0
    
    for idx in range(n_samples):
        seq = val_seqs[idx]
        weather_seq = weather_data[idx]
        
        # 빈 시퀀스 스킵
        if np.sum(seq) == 0:
            continue
        
        # 실제 예측 가능한 프레임 수 계산
        seq_len = seq.shape[0]
        available_frames = seq_len - input_len
        actual_num_to_predict = min(num_to_predict, available_frames)
        
        if actual_num_to_predict <= 0:
            continue  # 예측할 프레임이 없으면 스킵
        
        valid_count += 1
        true_frames = seq[input_len:input_len + actual_num_to_predict]
        all_predicted_frames = np.copy(seq[:input_len])
        
        # 자기회귀 예측
        for i in range(actual_num_to_predict):
            current_image_input = np.expand_dims(all_predicted_frames[-input_len:], axis=0)
            
            # 기상 데이터: 항상 처음 input_len 개 사용 (시간 축 크기 고정)
            # 기상 데이터는 시퀀스 전체에 대해 동일하게 적용되므로 첫 input_len개 사용
            current_weather_input = np.expand_dims(weather_seq[:input_len], axis=0)
            
            # float32로 명시적 변환
            current_image_input = current_image_input.astype(np.float32)
            current_weather_input = current_weather_input.astype(np.float32)
            
            new_prediction = model.predict({
                'image_input': current_image_input,
                'weather_input': current_weather_input
            }, verbose=0)
            
            predicted_frame = np.squeeze(new_prediction, axis=0)[-1]
            all_predicted_frames = np.concatenate(
                (all_predicted_frames, np.expand_dims(predicted_frame, axis=0)), axis=0
            )
            
            # 프레임별 메트릭 계산 (실제 예측 가능한 프레임만)
            if i < len(true_frames):
                frame_iou = calculate_iou(true_frames[i], predicted_frame, threshold)
                frame_dice = calculate_dice(true_frames[i], predicted_frame, threshold)
                # 프레임 인덱스는 1부터 시작하므로 i+1 사용
                frame_key = f'frame_{i+1}'
                if frame_key in metrics_per_frame:
                    metrics_per_frame[frame_key]['iou'].append(frame_iou)
                    metrics_per_frame[frame_key]['dice'].append(frame_dice)
        
        # 전체 예측 시퀀스 메트릭
        predicted_frames = all_predicted_frames[-actual_num_to_predict:]
        
        avg_iou = np.mean([calculate_iou(true_frames[i], predicted_frames[i], threshold) 
                          for i in range(actual_num_to_predict)])
        avg_dice = np.mean([calculate_dice(true_frames[i], predicted_frames[i], threshold) 
                           for i in range(actual_num_to_predict)])
        avg_acc = np.mean([calculate_pixel_accuracy(true_frames[i], predicted_frames[i], threshold) 
                          for i in range(actual_num_to_predict)])
        
        prec_rec_f1 = [calculate_precision_recall_f1(true_frames[i], predicted_frames[i], threshold) 
                       for i in range(actual_num_to_predict)]
        avg_prec = np.mean([x[0] for x in prec_rec_f1])
        avg_rec = np.mean([x[1] for x in prec_rec_f1])
        avg_f1 = np.mean([x[2] for x in prec_rec_f1])
        
        metrics_per_sample['iou'].append(avg_iou)
        metrics_per_sample['dice'].append(avg_dice)
        metrics_per_sample['pixel_accuracy'].append(avg_acc)
        metrics_per_sample['precision'].append(avg_prec)
        metrics_per_sample['recall'].append(avg_rec)
        metrics_per_sample['f1'].append(avg_f1)
        
        if valid_count % 10 == 0:
            print(f"  진행: {valid_count}개 유효 시퀀스 처리")
    
    # 결과 정리
    results = {
        'n_valid_samples': valid_count,
        'input_len': input_len,
        'num_to_predict': num_to_predict,
        'threshold': threshold,
        'overall_metrics': {},
        'per_frame_metrics': {}
    }
    
    for metric_name, values in metrics_per_sample.items():
        if len(values) > 0:
            results['overall_metrics'][metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
    
    for frame_name, frame_metrics in metrics_per_frame.items():
        results['per_frame_metrics'][frame_name] = {}
        for metric_name, values in frame_metrics.items():
            if len(values) > 0:
                results['per_frame_metrics'][frame_name][metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values))
                }
    
    return results


def print_results(results: dict, title: str = "평가 결과"):
    """평가 결과 출력"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)
    
    if 'metrics' in results:
        print(f"\n전체 샘플 수: {results['n_samples']}")
        if 'n_valid_samples' in results:
            print(f"유효 샘플 수: {results['n_valid_samples']} (빈 샘플 제외)")
        print(f"임계값: {results['threshold']}")
        print("\n[메트릭 요약]")
        print("-" * 40)
        for metric_name, stats in results['metrics'].items():
            print(f"  {metric_name:18s}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    if 'overall_metrics' in results:
        print(f"\n유효 샘플 수: {results['n_valid_samples']}")
        print(f"입력 프레임: {results['input_len']}, 예측 프레임: {results['num_to_predict']}")
        print("\n[전체 메트릭 요약]")
        print("-" * 40)
        for metric_name, stats in results['overall_metrics'].items():
            print(f"  {metric_name:18s}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        print("\n[프레임별 메트릭]")
        print("-" * 40)
        for frame_name, frame_metrics in results['per_frame_metrics'].items():
            print(f"  {frame_name}:")
            for metric_name, stats in frame_metrics.items():
                print(f"    {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    print("=" * 60)


def save_results(results: dict, save_path: str):
    """평가 결과를 JSON 파일로 저장"""
    # values 리스트 제거 (파일 크기 축소)
    results_to_save = results.copy()
    if 'metrics' in results_to_save:
        for metric_name in results_to_save['metrics']:
            if 'values' in results_to_save['metrics'][metric_name]:
                del results_to_save['metrics'][metric_name]['values']
    
    results_to_save['timestamp'] = datetime.now().isoformat()
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, indent=2, ensure_ascii=False)
    
    print(f"\n결과 저장됨: {save_path}")


# ===============================
# 시각화 함수
# ===============================

def visualize_sample_prediction(x_img: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                                  sample_idx: int, metrics: dict, save_path: str = None):
    """
    개별 샘플의 예측 결과와 메트릭 시각화
    
    Parameters
    ----------
    x_img : np.ndarray
        입력 이미지 시퀀스 (time, height, width, channels)
    y_true : np.ndarray
        Ground truth 시퀀스
    y_pred : np.ndarray
        예측 시퀀스
    sample_idx : int
        샘플 인덱스
    metrics : dict
        해당 샘플의 메트릭
    save_path : str, optional
        저장 경로
    """
    n_frames = y_true.shape[0]
    
    fig = plt.figure(figsize=(4 * n_frames + 2, 10))
    
    # GridSpec으로 레이아웃 구성
    gs = fig.add_gridspec(3, n_frames + 1, width_ratios=[1] * n_frames + [0.5])
    
    # Row 0: Ground Truth
    for i in range(n_frames):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(y_true[i].squeeze(), cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'GT Frame {i+1}', fontsize=10)
        ax.axis('off')
    
    # Row 1: Prediction
    for i in range(n_frames):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(y_pred[i].squeeze(), cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'Pred Frame {i+1}', fontsize=10)
        ax.axis('off')
    
    # Row 2: Difference (|GT - Pred|)
    for i in range(n_frames):
        ax = fig.add_subplot(gs[2, i])
        diff = np.abs(y_true[i].squeeze() - y_pred[i].squeeze())
        ax.imshow(diff, cmap='hot', vmin=0, vmax=1)
        ax.set_title(f'Diff Frame {i+1}', fontsize=10)
        ax.axis('off')
    
    # 메트릭 표시 (오른쪽 패널)
    ax_metrics = fig.add_subplot(gs[:, -1])
    ax_metrics.axis('off')
    
    metrics_text = f"Sample {sample_idx}\n"
    metrics_text += "-" * 15 + "\n"
    for key, value in metrics.items():
        if isinstance(value, float):
            metrics_text += f"{key}: {value:.4f}\n"
    
    ax_metrics.text(0.1, 0.5, metrics_text, transform=ax_metrics.transAxes,
                   fontsize=11, verticalalignment='center', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle(f'Sample {sample_idx}: Prediction Evaluation', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_metrics_summary(results: dict, save_path: str = None):
    """
    전체 메트릭 요약 시각화 (박스 플롯 + 히스토그램)
    
    Parameters
    ----------
    results : dict
        평가 결과
    save_path : str, optional
        저장 경로
    """
    metrics = results.get('metrics', {})
    if not metrics:
        print("메트릭 데이터가 없습니다.")
        return
    
    # 시각화할 메트릭 선택 (값이 있는 것만)
    valid_metrics = {k: v for k, v in metrics.items() 
                     if 'values' in v and len(v['values']) > 0}
    
    n_metrics = len(valid_metrics)
    if n_metrics == 0:
        print("시각화할 메트릭이 없습니다.")
        return
    
    fig, axes = plt.subplots(2, n_metrics, figsize=(4 * n_metrics, 8))
    
    for i, (metric_name, metric_data) in enumerate(valid_metrics.items()):
        values = np.array(metric_data['values'])
        
        # 박스 플롯
        ax_box = axes[0, i] if n_metrics > 1 else axes[0]
        ax_box.boxplot(values, vert=True)
        ax_box.set_title(f'{metric_name}', fontsize=12)
        ax_box.set_ylabel('Value')
        ax_box.axhline(y=metric_data['mean'], color='r', linestyle='--', label=f"mean={metric_data['mean']:.4f}")
        ax_box.legend(fontsize=8)
        
        # 히스토그램
        ax_hist = axes[1, i] if n_metrics > 1 else axes[1]
        ax_hist.hist(values, bins=20, edgecolor='black', alpha=0.7)
        ax_hist.axvline(x=metric_data['mean'], color='r', linestyle='--', linewidth=2)
        ax_hist.set_xlabel('Value')
        ax_hist.set_ylabel('Count')
        ax_hist.set_title(f'Distribution (std={metric_data["std"]:.4f})')
    
    plt.suptitle('Evaluation Metrics Summary', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"메트릭 요약 시각화 저장됨: {save_path}")
    else:
        plt.show()


def visualize_valid_vs_empty_analysis(results: dict, save_path: str = None):
    """
    유효 샘플 vs 빈 샘플 분석 시각화
    
    Parameters
    ----------
    results : dict
        평가 결과
    save_path : str, optional
        저장 경로
    """
    metrics = results.get('metrics', {})
    if 'iou' not in metrics or 'values' not in metrics['iou']:
        print("IoU 데이터가 없습니다.")
        return
    
    iou_values = np.array(metrics['iou']['values'])
    precision_values = np.array(metrics['precision']['values'])
    
    # 빈 샘플 식별 (IoU=1.0 이고 Precision=0인 경우)
    empty_mask = (iou_values == 1.0) & (precision_values == 0.0)
    valid_mask = ~empty_mask
    
    n_empty = np.sum(empty_mask)
    n_valid = np.sum(valid_mask)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. 파이 차트: 빈 샘플 vs 유효 샘플
    ax1 = axes[0]
    ax1.pie([n_valid, n_empty], labels=[f'Valid ({n_valid})', f'Empty ({n_empty})'],
            autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], explode=[0.05, 0])
    ax1.set_title('Valid vs Empty Samples')
    
    # 2. 유효 샘플만의 IoU 분포
    ax2 = axes[1]
    valid_iou = iou_values[valid_mask]
    if len(valid_iou) > 0:
        ax2.hist(valid_iou, bins=20, edgecolor='black', alpha=0.7, color='#3498db')
        ax2.axvline(x=np.mean(valid_iou), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(valid_iou):.4f}')
        ax2.set_xlabel('IoU')
        ax2.set_ylabel('Count')
        ax2.set_title(f'IoU Distribution (Valid Samples Only, n={n_valid})')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No valid samples', ha='center', va='center')
    
    # 3. 샘플별 IoU 스캐터 플롯
    ax3 = axes[2]
    sample_indices = np.arange(len(iou_values))
    ax3.scatter(sample_indices[valid_mask], iou_values[valid_mask], 
               c='#2ecc71', label='Valid', alpha=0.7, s=50)
    ax3.scatter(sample_indices[empty_mask], iou_values[empty_mask], 
               c='#e74c3c', label='Empty', alpha=0.7, s=50, marker='x')
    ax3.set_xlabel('Sample Index')
    ax3.set_ylabel('IoU')
    ax3.set_title('IoU by Sample')
    ax3.legend()
    ax3.set_ylim(-0.05, 1.05)
    
    plt.suptitle('Valid vs Empty Sample Analysis', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"유효/빈 샘플 분석 시각화 저장됨: {save_path}")
    else:
        plt.show()
    
    # 유효 샘플만의 실제 메트릭 출력
    if n_valid > 0:
        print(f"\n[유효 샘플만의 실제 메트릭] (n={n_valid})")
        print("-" * 40)
        for metric_name, metric_data in metrics.items():
            if 'values' in metric_data:
                valid_values = np.array(metric_data['values'])[valid_mask]
                print(f"  {metric_name:18s}: {np.mean(valid_values):.4f} ± {np.std(valid_values):.4f}")


def visualize_worst_samples(x_val: np.ndarray, y_val: np.ndarray, y_pred: np.ndarray,
                            results: dict, save_dir: str, n_samples: int = 5):
    """
    성능이 가장 낮은 샘플들 시각화
    
    Parameters
    ----------
    x_val : np.ndarray
        입력 데이터
    y_val : np.ndarray
        Ground truth
    y_pred : np.ndarray
        예측 결과
    results : dict
        평가 결과
    save_dir : str
        저장 디렉토리
    n_samples : int
        시각화할 샘플 수
    """
    metrics = results.get('metrics', {})
    if 'iou' not in metrics or 'values' not in metrics['iou']:
        return
    
    iou_values = np.array(metrics['iou']['values'])
    precision_values = np.array(metrics['precision']['values'])
    
    # 빈 샘플 제외
    empty_mask = (iou_values == 1.0) & (precision_values == 0.0)
    valid_indices = np.where(~empty_mask)[0]
    
    if len(valid_indices) == 0:
        print("유효한 샘플이 없습니다.")
        return
    
    # IoU 기준으로 가장 낮은 샘플들 선택
    valid_iou = iou_values[valid_indices]
    worst_indices_in_valid = np.argsort(valid_iou)[:n_samples]
    worst_indices = valid_indices[worst_indices_in_valid]
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n성능 최하위 {len(worst_indices)}개 샘플 시각화 중...")
    
    for rank, idx in enumerate(worst_indices):
        sample_metrics = {
            'iou': metrics['iou']['values'][idx],
            'dice': metrics['dice']['values'][idx],
            'precision': metrics['precision']['values'][idx],
            'recall': metrics['recall']['values'][idx],
        }
        
        save_path = os.path.join(save_dir, f"worst_{rank+1}_sample_{idx}.png")
        visualize_sample_prediction(
            x_val[idx], y_val[idx], y_pred[idx],
            sample_idx=idx, metrics=sample_metrics, save_path=save_path
        )
    
    print(f"  최하위 샘플 시각화 저장됨: {save_dir}")


def visualize_best_samples(x_val: np.ndarray, y_val: np.ndarray, y_pred: np.ndarray,
                           results: dict, save_dir: str, n_samples: int = 5):
    """
    성능이 가장 높은 샘플들 시각화 (유효 샘플 중)
    """
    metrics = results.get('metrics', {})
    if 'iou' not in metrics or 'values' not in metrics['iou']:
        return
    
    iou_values = np.array(metrics['iou']['values'])
    precision_values = np.array(metrics['precision']['values'])
    
    # 빈 샘플 제외
    empty_mask = (iou_values == 1.0) & (precision_values == 0.0)
    valid_indices = np.where(~empty_mask)[0]
    
    if len(valid_indices) == 0:
        print("유효한 샘플이 없습니다.")
        return
    
    # IoU 기준으로 가장 높은 샘플들 선택
    valid_iou = iou_values[valid_indices]
    best_indices_in_valid = np.argsort(valid_iou)[-n_samples:][::-1]
    best_indices = valid_indices[best_indices_in_valid]
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n성능 최상위 {len(best_indices)}개 샘플 시각화 중...")
    
    for rank, idx in enumerate(best_indices):
        sample_metrics = {
            'iou': metrics['iou']['values'][idx],
            'dice': metrics['dice']['values'][idx],
            'precision': metrics['precision']['values'][idx],
            'recall': metrics['recall']['values'][idx],
        }
        
        save_path = os.path.join(save_dir, f"best_{rank+1}_sample_{idx}.png")
        visualize_sample_prediction(
            x_val[idx], y_val[idx], y_pred[idx],
            sample_idx=idx, metrics=sample_metrics, save_path=save_path
        )
    
    print(f"  최상위 샘플 시각화 저장됨: {save_dir}")


def visualize_random_samples(x_val: np.ndarray, y_val: np.ndarray, y_pred: np.ndarray,
                             results: dict, save_dir: str, n_samples: int = 5):
    """
    랜덤 샘플들 시각화 (유효 샘플 중)
    """
    metrics = results.get('metrics', {})
    if 'iou' not in metrics or 'values' not in metrics['iou']:
        return
    
    iou_values = np.array(metrics['iou']['values'])
    precision_values = np.array(metrics['precision']['values'])
    
    # 빈 샘플 제외
    empty_mask = (iou_values == 1.0) & (precision_values == 0.0)
    valid_indices = np.where(~empty_mask)[0]
    
    if len(valid_indices) == 0:
        print("유효한 샘플이 없습니다.")
        return
    
    # 랜덤 선택
    n_to_select = min(n_samples, len(valid_indices))
    random_indices = np.random.choice(valid_indices, size=n_to_select, replace=False)
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n랜덤 {len(random_indices)}개 샘플 시각화 중...")
    
    for rank, idx in enumerate(random_indices):
        sample_metrics = {
            'iou': metrics['iou']['values'][idx],
            'dice': metrics['dice']['values'][idx],
            'precision': metrics['precision']['values'][idx],
            'recall': metrics['recall']['values'][idx],
        }
        
        save_path = os.path.join(save_dir, f"random_{rank+1}_sample_{idx}.png")
        visualize_sample_prediction(
            x_val[idx], y_val[idx], y_pred[idx],
            sample_idx=idx, metrics=sample_metrics, save_path=save_path
        )
    
    print(f"  랜덤 샘플 시각화 저장됨: {save_dir}")


# ===============================
# 메인 실행
# ===============================

def parse_args():
    parser = argparse.ArgumentParser(description="ConvLSTM 모델 평가")
    parser.add_argument("--data_path", type=str, default="./data/data.npy",
                        help="시퀀스 데이터 경로")
    parser.add_argument("--weather_path", type=str, default="./data/weather_ex.csv",
                        help="기상 데이터 경로")
    parser.add_argument("--weight_path", type=str, default="./weights/pred_with_weather.weights.h5",
                        help="모델 가중치 경로")
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="결과 저장 디렉토리")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="이진화 임계값")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="배치 크기")
    parser.add_argument("--filters", type=int, default=16,
                        help="ConvLSTM 필터 수")
    parser.add_argument("--eval_mode", type=str, default="both",
                        choices=["standard", "autoregressive", "both"],
                        help="평가 모드")
    parser.add_argument("--input_len", type=int, default=3,
                        help="자기회귀 평가 시 입력 프레임 수")
    parser.add_argument("--num_to_predict", type=int, default=2,
                        help="자기회귀 평가 시 예측 프레임 수")
    parser.add_argument("--visualize", action="store_true",
                        help="평가 결과 시각화 저장")
    parser.add_argument("--n_worst", type=int, default=5,
                        help="시각화할 최하위 샘플 수")
    parser.add_argument("--n_best", type=int, default=5,
                        help="시각화할 최상위 샘플 수")
    parser.add_argument("--n_random", type=int, default=5,
                        help="시각화할 랜덤 샘플 수")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "=" * 60)
    print(" ConvLSTM 모델 평가 시작")
    print("=" * 60)
    print(f"\n데이터 경로: {args.data_path}")
    print(f"가중치 경로: {args.weight_path}")
    print(f"평가 모드: {args.eval_mode}")
    
    # 결과 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # -------------------------------
    # 1. 데이터 로드
    # -------------------------------
    print("\n[1/4] 데이터 로드 중...")
    
    train_data, val_data = load_and_split_data(data_path=args.data_path)
    x_train_img, y_train_img = create_shifted_frames(train_data)
    x_val_img, y_val_img = create_shifted_frames(val_data)
    
    # 기상 데이터
    raw_weather_data = pd.read_csv(args.weather_path)
    weather_data = raw_weather_data.iloc[:6, [2, 3, 4, 5, 6, 7]]
    
    weather_min, weather_max = weather_data.min(), weather_data.max()
    weather_data_scaled = (weather_data - weather_min) / (weather_max - weather_min)
    
    DURATIONS = [1, 2, 2, 1]
    compressed_weather_data = compress_weather_data(weather_data_scaled, durations=DURATIONS)
    
    # shifted 후 이미지 시퀀스 길이에 맞춰 기상 데이터도 마지막 시점 제외
    # create_shifted_frames와 동일하게 처리: x는 0~n-2
    compressed_weather_data_x = compressed_weather_data[:-1]  # 마지막 시점 제외 (x에 맞춤)
    
    x_val_weather = np.tile(compressed_weather_data_x, (x_val_img.shape[0], 1, 1))
    
    print(f"  Validation 데이터: {x_val_img.shape}")
    print(f"  기상 데이터: {x_val_weather.shape}")
    
    # -------------------------------
    # 2. 모델 로드
    # -------------------------------
    print("\n[2/4] 모델 로드 중...")
    
    model = build_model(
        input_img_shape=x_val_img.shape[2:],
        input_weather_dim=x_val_weather.shape[-1],
        filters=args.filters
    )
    
    model.load_weights(args.weight_path)
    print(f"  가중치 로드 완료: {args.weight_path}")
    
    # -------------------------------
    # 3. 평가 수행
    # -------------------------------
    all_results = {}
    
    # 예측 결과 저장용 (시각화에 사용)
    y_pred_all = None
    
    if args.eval_mode in ["standard", "both"]:
        print("\n[3/4] 표준 평가 수행 중...")
        
        # 예측 수행 (시각화를 위해 결과 저장)
        y_pred_all = model.predict(
            {'image_input': x_val_img, 'weather_input': x_val_weather},
            batch_size=args.batch_size,
            verbose=0
        )
        
        standard_results = evaluate_model(
            model, x_val_img, x_val_weather, y_val_img,
            threshold=args.threshold,
            batch_size=args.batch_size
        )
        all_results['standard'] = standard_results
        print_results(standard_results, "표준 평가 결과")
    
    if args.eval_mode in ["autoregressive", "both"]:
        print("\n[3/4] 자기회귀 평가 수행 중...")
        
        # 자기회귀 평가용 모델: 고정된 시퀀스 길이로 새로 빌드
        ar_model = build_model(
            input_img_shape=x_val_img.shape[2:],
            input_weather_dim=x_val_weather.shape[-1],
            filters=args.filters,
            seq_len=args.input_len  # 시퀀스 길이 고정
        )
        ar_model.load_weights(args.weight_path)
        print(f"  자기회귀용 모델 로드 완료 (seq_len={args.input_len})")
        
        # 자기회귀 평가용 기상 데이터
        ar_weather = np.tile(compressed_weather_data, (val_data.shape[0], 1, 1))
        
        ar_results = evaluate_autoregressive(
            ar_model, val_data, ar_weather,
            input_len=args.input_len,
            num_to_predict=args.num_to_predict,
            threshold=args.threshold
        )
        all_results['autoregressive'] = ar_results
        print_results(ar_results, "자기회귀 평가 결과")
    
    # -------------------------------
    # 4. 결과 저장
    # -------------------------------
    print("\n[4/4] 결과 저장 중...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(result_dir, exist_ok=True)
    
    result_path = os.path.join(result_dir, "eval_results.json")
    save_results(all_results, result_path)
    
    # -------------------------------
    # 5. 시각화 (옵션)
    # -------------------------------
    if args.visualize and 'standard' in all_results and y_pred_all is not None:
        print("\n[5/5] 시각화 생성 중...")
        
        vis_dir = os.path.join(result_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # 메트릭 요약 시각화
        summary_path = os.path.join(vis_dir, "metrics_summary.png")
        visualize_metrics_summary(all_results['standard'], save_path=summary_path)
        
        # 유효/빈 샘플 분석
        analysis_path = os.path.join(vis_dir, "valid_vs_empty_analysis.png")
        visualize_valid_vs_empty_analysis(all_results['standard'], save_path=analysis_path)
        
        # 최하위 샘플 시각화
        worst_dir = os.path.join(vis_dir, "worst_samples")
        visualize_worst_samples(
            x_val_img, y_val_img, y_pred_all,
            all_results['standard'], worst_dir, n_samples=args.n_worst
        )
        
        # 최상위 샘플 시각화 (유효 샘플 중)
        best_dir = os.path.join(vis_dir, "best_samples")
        visualize_best_samples(
            x_val_img, y_val_img, y_pred_all,
            all_results['standard'], best_dir, n_samples=args.n_best
        )
        
        # 랜덤 샘플 시각화 (유효 샘플 중)
        random_dir = os.path.join(vis_dir, "random_samples")
        visualize_random_samples(
            x_val_img, y_val_img, y_pred_all,
            all_results['standard'], random_dir, n_samples=args.n_random
        )
        
        print(f"\n시각화 저장 완료: {vis_dir}")
    
    print(f"\n결과 저장 완료: {result_dir}")
    print("\n평가 완료!")


if __name__ == "__main__":
    main()

