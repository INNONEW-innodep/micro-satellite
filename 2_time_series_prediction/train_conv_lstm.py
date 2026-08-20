import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# TF import 전에 XLA 비활성화
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
tf.config.optimizer.set_jit(False)
import keras
from keras import layers

# Optional: mixed precision (메모리 절감)
# 일부 연산을 float16으로 수행하고, 중요한 연산(예: loss 계산, 가중치 누적)은 float32로 수행
USE_MIXED_PRECISION = True

if USE_MIXED_PRECISION:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("mixed_float16")  # 자동 loss scaling 처리


from data_preprocess import load_and_split_data, create_shifted_frames, compress_weather_data
# from visualize import ImagePredictionCallback

SEQ_DATA_PATH = "./data/busan/data.npy"
WEATHER_DATA_PATH = "./data/weather_20200218_20200414.csv"
SAVE_WEIGHT_DIR = "./weights"
SAVE_WEIGHT_PATH = "busan_model.weights.h5"


# -------------------------------
# 1. 데이터 준비
# -------------------------------

# 위성 이미지 시퀀스 데이터 로드, train/val 분할
train_data, val_data = load_and_split_data(data_path=SEQ_DATA_PATH)

# sliding window 형식의 학습을 위해 x/y 분할.
# 1시퀀스: 5개 이미지 >> x: 1~4번 / y: 2~5번
x_train_img, y_train_img = create_shifted_frames(train_data)
x_val_img, y_val_img = create_shifted_frames(val_data)

# 기상 데이터 로드
raw_weather_data = pd.read_csv(WEATHER_DATA_PATH)
weather_data = raw_weather_data.iloc[:6, [2, 3, 4, 5, 6, 7]]

# 기상 데이터 정규화
weather_min, weather_max = weather_data.min(), weather_data.max()
weather_data_scaled = (weather_data - weather_min) / (weather_max - weather_min)

# 원본 이미지 시퀀스 길이 확인 (shifted 전)
original_seq_len = train_data.shape[1]  # 예: 4개 프레임
# shifted 후 이미지 시퀀스 길이 확인
shifted_seq_len = x_train_img.shape[1]  # 예: 3개 프레임

# 기상 데이터를 원본 시퀀스 길이에 맞춰 압축
# 예: 6일 데이터를 4개 간격으로 재구성
DURATIONS = [1, 2, 2, 1] 
compressed_weather_data = compress_weather_data(weather_data_scaled, durations=DURATIONS)

# shifted 후 이미지 시퀀스 길이에 맞춰 기상 데이터도 마지막 시점 제외
# create_shifted_frames와 동일하게 처리: x는 0~n-2, y는 1~n-1
compressed_weather_data_x = compressed_weather_data[:-1]  # 마지막 시점 제외 (x에 맞춤)

# 위성 이미지 시퀀스 데이터 수에 맞춰 기상 데이터 준비
x_train_weather = np.tile(compressed_weather_data_x, (x_train_img.shape[0], 1, 1))
x_val_weather = np.tile(compressed_weather_data_x, (x_val_img.shape[0], 1, 1))



# 최종 데이터셋 확인
print("x_train_img:", x_train_img.shape, "y_train_img:", y_train_img.shape)
print("x_train_weather:", x_train_weather.shape)
print("x_val_img  :", x_val_img.shape, "y_val_img  :", y_val_img.shape)
print("x_val_weather:", x_val_weather.shape)


# -------------------------------
# 2. 모델 구축 (인코더-디코더)
# -------------------------------
strategy = tf.distribute.MirroredStrategy()
num_replicas = strategy.num_replicas_in_sync
batch_size = 1 * num_replicas

with strategy.scope():
    # 1. 입력 레이어 정의
    image_input = layers.Input(shape=(None, *x_train_img.shape[2:]), name='image_input')
    weather_input = layers.Input(shape=(None, x_train_weather.shape[-1]), name='weather_input')

    # 2. 인코더: Conv-LSTM으로 이미지 다운샘플링 (512x512 -> 256x256 -> 128x128)
    f = 16
    x_img = layers.ConvLSTM2D(f, (5, 5), padding="same", strides=(2, 2), return_sequences=True, activation="relu")(image_input)
    x_img = layers.LayerNormalization()(x_img)
    x_img = layers.ConvLSTM2D(f, (3, 3), padding="same", strides=(2, 2), return_sequences=True, activation="relu")(x_img)
    x_img = layers.LayerNormalization()(x_img)

    # 3. 날씨 데이터 결합
    h, w = x_img.shape[2], x_img.shape[3]
    weather_tiled = layers.TimeDistributed(layers.Dense(h * w * 1))(weather_input)
    weather_reshaped = layers.Reshape((x_img.shape[1], h, w, 1))(weather_tiled)
    combined_features = layers.Concatenate(axis=-1)([x_img, weather_reshaped])

    # 4. 디코더: Conv3DTranspose로 이미지 업샘플링 (128x128 -> 512x512)
    x_decode = layers.Conv3DTranspose(f, (3, 3, 3), padding="same", strides=(1, 2, 2), activation="relu")(combined_features)
    x_decode = layers.LayerNormalization()(x_decode)
    x_decode = layers.Conv3DTranspose(f, (3, 3, 3), padding="same", strides=(1, 2, 2), activation="relu")(x_decode)
    x_decode = layers.LayerNormalization()(x_decode)

    # 최종 예측 (결과는 원본 이미지와 동일한 크기)
    out = layers.Conv3D(1, (3, 3, 3), padding="same", activation="sigmoid")(x_decode)
    
    # 모델 정의 및 컴파일
    model = keras.models.Model(inputs=[image_input, weather_input], outputs=out)
    
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.binary_crossentropy,
        run_eagerly=True,
    )

model.summary()


# -------------------------------
# 3. 모델 학습
# -------------------------------
try:
    model.fit(
        {'image_input': x_train_img, 'weather_input': x_train_weather},
        y_train_img,
        batch_size=batch_size,
        epochs=100,
        validation_data=({'image_input': x_val_img, 'weather_input': x_val_weather}, y_val_img),
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5),
        ],
    )
except tf.errors.ResourceExhaustedError as e:
    print("ResourceExhaustedError (OOM) detected. Try reducing batch_size, filters, or enable mixed precision.")
    raise
except Exception as e:
    print("Training raised exception:", type(e), e)
    raise

# weight 파일 저장
os.makedirs(SAVE_WEIGHT_DIR, exist_ok=True)
model.save_weights(os.path.join(SAVE_WEIGHT_DIR, SAVE_WEIGHT_PATH))