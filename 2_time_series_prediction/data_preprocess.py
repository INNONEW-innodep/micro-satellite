"""
Data preprocess for training conv-lstm model 
"""

import numpy as np
import pandas as pd


def load_and_split_data(data_path: str, split_ratio: float = 0.9, seed: int = 42):
    """
    위성 이미지 시퀀스 데이터 로드 후 train/val으로 분할
    
    Parameters
    ----------
    path : str
        .npy 데이터 경로
    split_ratio : float
        학습(train) 데이터 비율 (default=0.9)
    seed : int
        랜덤 시드 (default=42)
    
    Returns
    -------
    train_dataset : np.ndarray
    val_dataset : np.ndarray
    """
    
    # 데이터 로드
    data = np.load(data_path)
    print(f"Sequence data: {data.shape}")  
    
    # 인덱스 셔플
    np.random.seed(seed)
    indexes = np.arange(data.shape[0])
    np.random.shuffle(indexes)
    
    # Split
    split_point = int(split_ratio * data.shape[0])
    train_index = indexes[:split_point]
    val_index = indexes[split_point:]
    
    train_data = data[train_index]
    val_data = data[val_index]
    
    print(f"train: {train_data.shape}, val: {val_data.shape}")
    return train_data, val_data


def create_shifted_frames(data):
    # data shape: (batch, time, height, width, channels)
    x = data[:, 0 : data.shape[1] - 1, :, :, :]
    y = data[:, 1 : data.shape[1], :, :, :]
    return x.astype(np.float32), y.astype(np.float32)


def compress_weather_data(weather_data: pd.DataFrame, durations: list[int]) -> np.ndarray:
    """
    일자별 기상 데이터를 시퀀스 데이터의 시점 간 기간에 맞춰 재구성 
    
    Parameters
    ----------
    weather_df : pd.DataFrame
        원본 기상 데이터 (행=날짜, 열=특징)
    durations : list[int]
        각 시퀀스 시점이 커버하는 기간 (일 단위)
    
    Returns
    -------
    compressed_weather_sequence : np.ndarray
        (시점 개수, feature+1) 크기의 배열
        마지막 열은 정규화된 기간 값
    """
    
    n_days = len(weather_data)
    total_duration = sum(durations)
    
    if total_duration != n_days:
        raise ValueError(
            f"durations 합({total_duration}) != 기상 데이터 일수({n_days})"
        )
    
    compressed = []
    start_idx = 0
    
    for d in durations:
        # 구간 평균 (기간 d일 동안의 기상 데이터 평균)
        group = weather_data.iloc[start_idx:start_idx + d, :].mean(axis=0).values
        # 기간 값 추가
        group = np.append(group, d)
        compressed.append(group)
        
        start_idx += d
    
    compressed = np.stack(compressed)
    
    # 기간 값 정규화
    max_duration = np.max(compressed[:, -1])
    compressed[:, -1] = compressed[:, -1] / max_duration
    
    return compressed