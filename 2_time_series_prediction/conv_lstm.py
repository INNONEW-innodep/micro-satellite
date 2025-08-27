import os
# TF import 전에 XLA 비활성화
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

# Optional: mixed precision (메모리 절감)
USE_MIXED_PRECISION = True

if USE_MIXED_PRECISION:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("mixed_float16")  # 자동으로 loss scaling 처리

# 데이터 로드
data = np.load("./data/data.npy")
indexes = np.arange(data.shape[0])
np.random.shuffle(indexes)
train_index = indexes[: int(0.9 * data.shape[0])]
val_index = indexes[int(0.9 * data.shape[0]) :]

train_dataset = data[train_index].astype(np.float32)
val_dataset = data[val_index].astype(np.float32)

print("raw train shape:", train_dataset.shape)
print("raw val   shape:", val_dataset.shape)

def create_shifted_frames(data):
    # data shape: (batch, time, H, W, C)
    x = data[:, 0 : data.shape[1] - 1, :, :, :]
    y = data[:, 1 : data.shape[1], :, :, :]
    return x.astype(np.float32), y.astype(np.float32)

x_train, y_train = create_shifted_frames(train_dataset)
x_val, y_val = create_shifted_frames(val_dataset)

print("x_train:", x_train.shape, "y_train:", y_train.shape)
print("x_val  :", x_val.shape,   "y_val  :", y_val.shape)

# -------------------------------
# Multi-GPU 전략 선언
# -------------------------------
strategy = tf.distribute.MirroredStrategy()
num_replicas = strategy.num_replicas_in_sync
print("Number of replicas (GPUs):", num_replicas)

# 안전한 batch_size 설정: per_replica=1 권장, 필요시 늘리세요.
per_replica_batch_size = 1
batch_size = per_replica_batch_size * num_replicas
print("Using global batch_size:", batch_size, "(per-replica:", per_replica_batch_size, ")")

with strategy.scope():
    inp = layers.Input(shape=(None, *x_train.shape[2:]))  # (T-1, H, W, C)

    # 모델: 필터 수를 필요시 더 낮춰 메모리 절약 (예: 8)
    f = 64  # 기본 16, 메모리 문제 시 8 또는 4로 줄이세요
    x = layers.ConvLSTM2D(f, (5, 5), padding="same", return_sequences=True, activation="relu")(inp)
    x = layers.LayerNormalization()(x)
    x = layers.ConvLSTM2D(f, (3, 3), padding="same", return_sequences=True, activation="relu")(x)
    x = layers.LayerNormalization()(x)
    x = layers.ConvLSTM2D(f, (1, 1), padding="same", return_sequences=True, activation="relu")(x)

    # Mixed precision 사용중이면 Conv3D의 계산은 float16, 하지만 출력은 float32로 캐스트해서 손실 계산 안정화
    out = layers.Conv3D(1, (3, 3, 3), padding="same")(x)
    out = layers.Activation("sigmoid", dtype="float32")(out)  # 마지막은 float32

    model = keras.models.Model(inp, out)

    # Compile: mixed precision이면 optimizer에 'Adam' 사용 가능 (loss scaling 자동)
    opt = keras.optimizers.Adam()
    model.compile(
        optimizer=opt,
        loss=keras.losses.binary_crossentropy,
        jit_compile=False,
        run_eagerly=False,
    )

# shape 체크: 모델 출력과 y_train 일치 여부
print("Model output shape (None for batch):", model.output_shape)
print("y_train shape (should match output excluding batch dim):", y_train.shape)

# Dummy call to build variables (이미 하셨지만 안전하게 한 번 더)
dummy_time = x_train.shape[1]
if dummy_time is None or dummy_time == 0:
    dummy_time = 1
dummy = tf.zeros((1, dummy_time, x_train.shape[2], x_train.shape[3], x_train.shape[4]), dtype=tf.float32)
_ = model(dummy, training=False)

# 콜백
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)

# === 주의: 먼저 작은 batch_size로 테스트 ===
try:
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=100,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping, reduce_lr],
    )
except tf.errors.ResourceExhaustedError as e:
    print("ResourceExhaustedError (OOM) detected. Try reducing batch_size, filters, or enable mixed precision.")
    raise
except Exception as e:
    print("Training raised exception:", type(e), e)
    raise

# 저장
weights_dir = "/workspace/weights"
os.makedirs(weights_dir, exist_ok=True)
model.save_weights(os.path.join(weights_dir, "pred_wb.weights.h5"))
