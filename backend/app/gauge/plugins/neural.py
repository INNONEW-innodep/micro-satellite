"""딥러닝 수위 예측기 플러그인 — GPU(PyTorch)로 학습, numpy 로 추론.

MLP · LSTM · GRU · TCN 네 구조를 같은 입력·같은 목표로 학습한다:

- 입력: 원점 직전 ``window`` 일의 수위(원점 수위 대비 편차, 학습 표준편차로 나눔),
  30·365일 이동평균 편차, 연중 조화항, (있으면) 강수 log1p·기온 편차.
- 목표: ``y[t+h] − y[t]``, h = 1..30 을 **한 번에**(30 출력). 그래서 ``fit`` 이 지평마다
  불려도 첫 호출에서만 학습하고 나머지는 캐시를 돌려준다(``shared_ref``).
- 추론: 가중치를 numpy 배열로 내보내 서버(CPU 슬림 이미지, torch 없음)에서도 같은 결과를 낸다.

학습은 ``GAUGE_FORECASTER_PLUGINS`` 로 등록한 뒤 torch 가 있는 환경에서
``scripts/train_gauge_models.py`` 를 돌린다. torch 가 없으면 학습만 실패하고(후보에서
제외), 이미 학습된 산출물의 추론은 numpy 만으로 동작한다.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from ..forecasters import (
    DailySeries,
    ForecasterInfo,
    GaugeForecaster,
    exog_features,
    rolling_mean,
    seasonal_features,
)

H_MAX = 30


# --- 특징 -------------------------------------------------------------------------


def _sequence_features(series: DailySeries, origins: np.ndarray, window: int, scale: float) -> np.ndarray:
    """(n, window, f) 시퀀스 특징. 각 스텝: 수위 편차, 강수 log1p, 기온 편차(있으면)."""

    origins = np.asarray(origins, dtype=int)
    idx = origins[:, None] - np.arange(window)[None, ::-1]  # 오래된 → 최근
    valid = idx >= 0
    idx_c = np.clip(idx, 0, series.n - 1)
    level = (series.filled[idx_c] - series.filled[origins][:, None]) / scale
    level[~valid] = 0.0
    channels = [level]
    if series.has_exog("rain_mm"):
        rain = np.log1p(np.maximum(np.nan_to_num(series.exog["rain_mm"], nan=0.0), 0.0))
        r = rain[idx_c]
        r[~valid] = 0.0
        channels.append(r / 3.0)
    if series.has_exog("temp_c"):
        temp = series.exog["temp_c"]
        temp = np.where(np.isnan(temp), np.nanmean(temp) if np.isfinite(np.nanmean(temp)) else 0.0, temp)
        tt = temp[idx_c]
        tt[~valid] = 0.0
        channels.append((tt - 12.0) / 10.0)
    return np.stack(channels, axis=-1).astype(np.float32)


def _static_features(series: DailySeries, origins: np.ndarray, scale: float) -> np.ndarray:
    """(n, s) 원점 정적 특징: 30·365일 이동평균 편차, 연중 조화항, 기상 요약."""

    origins = np.asarray(origins, dtype=int)
    cur = series.filled[origins]
    r30 = rolling_mean(series.filled, 30, origins)
    r365 = rolling_mean(series.filled, 365, origins)
    r30 = np.where(np.isnan(r30), cur, r30)
    r365 = np.where(np.isnan(r365), cur, r365)
    cols = [(cur - r30) / scale, (cur - r365) / scale]
    season = seasonal_features(series.doy[origins])
    cols.extend(season.T)
    extra = exog_features(series, origins)
    if extra is not None:
        extra = np.nan_to_num(extra, nan=0.0)
        cols.extend((extra / np.array([3.0, 4.0, 5.0, 6.0, 10.0, 5.0][: extra.shape[1]])).T)
    return np.stack(cols, axis=1).astype(np.float32)


def _targets(series: DailySeries, origins: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """(n, H_MAX) 목표(편차/scale)와 마스크."""

    origins = np.asarray(origins, dtype=int)
    idx = origins[:, None] + np.arange(1, H_MAX + 1)[None, :]
    inside = idx < series.n
    idx_c = np.clip(idx, 0, series.n - 1)
    y = series.y[idx_c]
    mask = inside & ~np.isnan(y)
    target = np.where(mask, (y - series.filled[origins][:, None]) / scale, 0.0)
    return target.astype(np.float32), mask


# --- numpy 추론 -----------------------------------------------------------------------


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _forward_mlp(w: Mapping[str, np.ndarray], seq: np.ndarray, static: np.ndarray) -> np.ndarray:
    x = np.concatenate([seq.reshape(seq.shape[0], -1), static], axis=1)
    h = _relu(x @ w["fc1.weight"].T + w["fc1.bias"])
    h = _relu(h @ w["fc2.weight"].T + w["fc2.bias"])
    return h @ w["head.weight"].T + w["head.bias"]


def _forward_rnn(w: Mapping[str, np.ndarray], seq: np.ndarray, static: np.ndarray, kind: str) -> np.ndarray:
    n, T, _f = seq.shape
    hidden = w["rnn.weight_hh_l0"].shape[1]
    h = np.zeros((n, hidden), dtype=np.float32)
    c = np.zeros((n, hidden), dtype=np.float32)
    w_ih, w_hh = w["rnn.weight_ih_l0"], w["rnn.weight_hh_l0"]
    b = w["rnn.bias_ih_l0"] + w["rnn.bias_hh_l0"]
    for t in range(T):
        x = seq[:, t, :]
        gates = x @ w_ih.T + h @ w_hh.T + b
        if kind == "lstm":
            i, f, g, o = np.split(gates, 4, axis=1)
            i, f, o = _sigmoid(i), _sigmoid(f), _sigmoid(o)
            g = np.tanh(g)
            c = f * c + i * g
            h = o * np.tanh(c)
        else:  # gru (PyTorch 규약: r, z, n)
            xr, xz, xn = np.split(x @ w_ih.T + w["rnn.bias_ih_l0"], 3, axis=1)
            hr, hz, hn = np.split(h @ w_hh.T + w["rnn.bias_hh_l0"], 3, axis=1)
            r = _sigmoid(xr + hr)
            z = _sigmoid(xz + hz)
            nn_ = np.tanh(xn + r * hn)
            h = (1 - z) * nn_ + z * h
    x = np.concatenate([h, static], axis=1)
    hh = _relu(x @ w["fc.weight"].T + w["fc.bias"])
    return hh @ w["head.weight"].T + w["head.bias"]


def _forward_tcn(w: Mapping[str, np.ndarray], seq: np.ndarray, static: np.ndarray, dilations: list[int]) -> np.ndarray:
    x = np.transpose(seq, (0, 2, 1))  # (n, C, T)
    for layer, d in enumerate(dilations):
        weight, bias = w[f"convs.{layer}.weight"], w[f"convs.{layer}.bias"]  # (Cout, Cin, K)
        k = weight.shape[2]
        n, cin, T = x.shape
        pad = (k - 1) * d
        xp = np.concatenate([np.zeros((n, cin, pad), dtype=x.dtype), x], axis=2)
        out = np.zeros((n, weight.shape[0], T), dtype=np.float32)
        for tap in range(k):
            sl = xp[:, :, tap * d : tap * d + T]
            out += np.einsum("nct,oc->not", sl, weight[:, :, tap])
        out += bias[None, :, None]
        x = _relu(out)
        if layer > 0 and x.shape == out.shape:
            pass
    last = x[:, :, -1]
    h = np.concatenate([last, static], axis=1)
    hh = _relu(h @ w["fc.weight"].T + w["fc.bias"])
    return hh @ w["head.weight"].T + w["head.bias"]


# --- 예측기 ----------------------------------------------------------------------------


class NeuralForecaster(GaugeForecaster):
    arch: str = "mlp"
    _id: str = "neural_mlp"
    _name: str = "MLP 신경망"
    _desc: str = "완전연결 신경망. 최근 window 일 수위·기상·계절 특징을 펼쳐 30개 지평 편차를 한 번에 학습(GPU)."
    _grid: tuple[Mapping[str, Any], ...] = ({"window": 30, "hidden": 64}, {"window": 60, "hidden": 64})

    def __init__(self) -> None:
        self._shared: dict[str, dict[str, Any]] = {}

    @property
    def info(self) -> ForecasterInfo:
        return ForecasterInfo(
            id=self._id, name=self._name, description=self._desc,
            hyperparameter_grid=self._grid, min_history_days=730,
        )

    # -- 캐시 키: 같은 자료·같은 원점 집합·같은 하이퍼파라미터면 한 번만 학습 --
    @staticmethod
    def _key(series: DailySeries, origins: np.ndarray, horizon: int, hparams: Mapping[str, Any], arch: str) -> str:
        """지평과 무관한 키: 폴드의 마지막 목표일(원점 최대 + h)이 같으면 같은 학습을 재사용한다."""

        h = hashlib.sha1()
        h.update(str(series.dates[0]).encode()); h.update(str(series.n).encode())
        h.update(np.ascontiguousarray(series.filled[::7]).tobytes()[:20000])
        origins = np.asarray(origins, dtype=int)
        last_target = int(origins.max()) + int(horizon)
        h.update(str(int(origins.min())).encode()); h.update(str(last_target).encode())
        h.update(json.dumps(dict(hparams), sort_keys=True).encode()); h.update(arch.encode())
        h.update(",".join(sorted(series.exog)).encode())
        return f"{arch}:{h.hexdigest()[:16]}"

    def fit(self, series, origins, horizon, hparams) -> dict[str, Any]:
        origins = self._usable_origins(series, origins, horizon)
        key = self._key(series, origins, horizon, hparams, self.arch)
        if key not in self._shared:
            self._shared[key] = self._train(series, origins, dict(hparams))
        return {"shared_ref": key, "horizon": int(horizon), "hparams": dict(hparams)}

    def predict(self, series, origins, horizon, params) -> np.ndarray:
        blob = self._shared.get(str(params.get("shared_ref")))
        origins = np.asarray(origins, dtype=int)
        out = np.full(origins.shape[0], np.nan, dtype=np.float64)
        if blob is None:
            return out
        w = {k: np.asarray(v, dtype=np.float32) for k, v in blob["weights"].items()}
        window, scale = int(blob["window"]), float(blob["scale"])
        ok = origins - window + 1 >= 0
        if not ok.any():
            return out
        seq = _sequence_features(series, origins[ok], window, scale)
        static = _static_features(series, origins[ok], scale)
        if seq.shape[2] != int(blob["seq_channels"]) or static.shape[1] != int(blob["static_dim"]):
            return out  # 학습 때와 외생 자료 구성이 다르면 예측하지 않는다
        if self.arch == "mlp":
            pred = _forward_mlp(w, seq, static)
        elif self.arch in ("lstm", "gru"):
            pred = _forward_rnn(w, seq, static, self.arch)
        else:
            pred = _forward_tcn(w, seq, static, list(blob["dilations"]))
        out[ok] = series.filled[origins[ok]] + pred[:, int(horizon) - 1] * scale
        return out

    @staticmethod
    def _encode(array: np.ndarray) -> dict[str, Any]:
        arr = np.asarray(array, dtype=np.float16)
        return {"shape": list(arr.shape), "f16": base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")}

    @staticmethod
    def _decode(blob: Any) -> np.ndarray:
        if isinstance(blob, Mapping) and "f16" in blob:
            return np.frombuffer(base64.b64decode(blob["f16"]), dtype=np.float16).astype(np.float32).reshape(blob["shape"])
        return np.asarray(blob, dtype=np.float32)

    def export_shared(self, refs: Iterable[str]) -> dict[str, Any]:
        out = {}
        for ref in refs:
            blob = self._shared.get(ref)
            if blob is None:
                continue
            out[ref] = {**{k: v for k, v in blob.items() if k != "weights"},
                        "weights": {k: self._encode(v) for k, v in blob["weights"].items()}}
        return out

    def import_shared(self, blobs: Mapping[str, Any]) -> None:
        for ref, blob in blobs.items():
            if isinstance(blob, Mapping) and str(ref).startswith(self.arch + ":") and "weights" in blob:
                self._shared[ref] = {**blob, "weights": {k: self._decode(v) for k, v in blob["weights"].items()}}

    # -- 학습 (torch, GPU 가 있으면 사용) --
    def _train(self, series: DailySeries, origins: np.ndarray, hparams: Mapping[str, Any]) -> dict[str, Any]:
        try:
            import torch
            from torch import nn
        except ImportError as exc:  # noqa: F841
            raise ValueError("torch 가 없어 신경망을 학습할 수 없습니다 (추론은 numpy 로 가능)") from None

        window = int(hparams.get("window", 30))
        hidden = int(hparams.get("hidden", 64))
        epochs = int(hparams.get("epochs", os.getenv("NEURAL_EPOCHS", "150")))
        lr = float(hparams.get("lr", 1e-3))
        origins = origins[origins - window + 1 >= 0]
        if origins.shape[0] < 200:
            raise ValueError("신경망 학습 원점이 부족합니다")
        scale = float(np.nanstd(series.y[origins])) or 1.0
        seq = _sequence_features(series, origins, window, scale)
        static = _static_features(series, origins, scale)
        target, mask = _targets(series, origins, scale)
        # 시간 순서 기준 마지막 10% 를 조기 종료용 검증으로
        n = origins.shape[0]
        cut = int(n * 0.9)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(int(hparams.get("seed", 7)))

        seq_dim, static_dim = seq.shape[2], static.shape[1]
        arch = self.arch

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                if arch == "mlp":
                    self.fc1 = nn.Linear(window * seq_dim + static_dim, hidden)
                    self.fc2 = nn.Linear(hidden, hidden)
                    self.head = nn.Linear(hidden, H_MAX)
                elif arch in ("lstm", "gru"):
                    cls = nn.LSTM if arch == "lstm" else nn.GRU
                    self.rnn = cls(seq_dim, hidden, batch_first=True)
                    self.fc = nn.Linear(hidden + static_dim, hidden)
                    self.head = nn.Linear(hidden, H_MAX)
                else:
                    self.dilations = [1, 2, 4, 8]
                    convs = []
                    cin = seq_dim
                    for d in self.dilations:
                        convs.append(nn.Conv1d(cin, hidden, kernel_size=3, dilation=d))
                        cin = hidden
                    self.convs = nn.ModuleList(convs)
                    self.fc = nn.Linear(hidden + static_dim, hidden)
                    self.head = nn.Linear(hidden, H_MAX)

            def forward(self, s, st):
                if arch == "mlp":
                    x = torch.cat([s.reshape(s.shape[0], -1), st], dim=1)
                    return self.head(torch.relu(self.fc2(torch.relu(self.fc1(x)))))
                if arch in ("lstm", "gru"):
                    out, _ = self.rnn(s)
                    h = out[:, -1, :]
                    return self.head(torch.relu(self.fc(torch.cat([h, st], dim=1))))
                x = s.transpose(1, 2)
                for conv, d in zip(self.convs, self.dilations):
                    pad = (conv.kernel_size[0] - 1) * d
                    x = torch.relu(conv(nn.functional.pad(x, (pad, 0))))
                h = x[:, :, -1]
                return self.head(torch.relu(self.fc(torch.cat([h, st], dim=1))))

        net = Net().to(device)
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
        S = torch.tensor(seq, device=device); ST = torch.tensor(static, device=device)
        Y = torch.tensor(target, device=device); M = torch.tensor(mask.astype(np.float32), device=device)
        idx_train = torch.arange(0, cut, device=device); idx_val = torch.arange(cut, n, device=device)
        best_state, best_loss, patience = None, float("inf"), 0
        batch = 256
        for epoch in range(epochs):
            net.train()
            perm = idx_train[torch.randperm(idx_train.shape[0], device=device)]
            for start in range(0, perm.shape[0], batch):
                b = perm[start : start + batch]
                opt.zero_grad()
                pred = net(S[b], ST[b])
                loss = (((pred - Y[b]) ** 2) * M[b]).sum() / M[b].sum().clamp(min=1.0)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            net.eval()
            with torch.no_grad():
                pv = net(S[idx_val], ST[idx_val])
                vloss = float((((pv - Y[idx_val]) ** 2) * M[idx_val]).sum() / M[idx_val].sum().clamp(min=1.0))
            if vloss < best_loss - 1e-6:
                best_loss, patience = vloss, 0
                best_state = {k: v.detach().cpu().numpy().copy() for k, v in net.state_dict().items()}
            else:
                patience += 1
                if patience >= 15:
                    break
        weights = best_state or {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
        return {
            "arch": arch, "window": window, "hidden": hidden, "scale": scale,
            "seq_channels": int(seq_dim), "static_dim": int(static_dim),
            "dilations": [1, 2, 4, 8] if arch == "tcn" else [],
            "epochs_run": int(epoch + 1), "val_loss": float(best_loss), "device": str(device),
            "weights": weights,
        }


class NeuralMLP(NeuralForecaster):
    arch = "mlp"; _id = "neural_mlp"; _name = "MLP 신경망 (GPU 학습)"
    _desc = "완전연결 신경망. 최근 30·60일 수위·기상·계절 특징을 펼쳐 30개 지평 편차를 한 번에 학습. torch(GPU)로 학습, numpy 로 추론."


class NeuralLSTM(NeuralForecaster):
    arch = "lstm"; _id = "neural_lstm"; _name = "LSTM 순환신경망 (GPU 학습)"
    _desc = "LSTM 이 최근 window 일 시퀀스(수위·강수·기온)를 읽고 30개 지평 편차를 출력. 은닉 64. torch(GPU) 학습, numpy 추론."
    _grid = ({"window": 30, "hidden": 64}, {"window": 60, "hidden": 64})


class NeuralGRU(NeuralForecaster):
    arch = "gru"; _id = "neural_gru"; _name = "GRU 순환신경망 (GPU 학습)"
    _desc = "GRU 순환신경망. LSTM 보다 가벼운 게이트 구조. 같은 입력·목표."
    _grid = ({"window": 30, "hidden": 64}, {"window": 60, "hidden": 64})


class NeuralTCN(NeuralForecaster):
    arch = "tcn"; _id = "neural_tcn"; _name = "TCN 시간합성곱 (GPU 학습)"
    _desc = "인과 팽창 합성곱(커널 3, 팽창 1·2·4·8) 4층. 순환 없이 긴 창을 병렬로 읽는다."
    _grid = ({"window": 30, "hidden": 32}, {"window": 60, "hidden": 32})
