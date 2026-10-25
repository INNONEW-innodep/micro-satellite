#!/usr/bin/env bash
# GPU 점검 — 이 서버에서 수체 탐지 이미지(wbms:0.13)가 GPU 로 돌 수 있는지 단계별로 확인하고 .env 의 WBMS_GPU 권장값을 낸다.
#
#   ./gpu_check.sh            전체 점검 (호스트 드라이버 → 컨테이너 툴킷 → 탐지 이미지 안 TensorFlow GPU → 합성곱 연산 실험)
#   ./gpu_check.sh --quick    합성곱 실험 생략
#
# 플랫폼(API·화면·수위 예측)은 GPU 를 쓰지 않으므로 이 점검과 무관하게 동작한다. 탐지만 GPU 가 안 되면 CPU 로 돈다(장면당 10분 내외).
# 2026-10-25: 현장(RTX PRO 5000 · 드라이버 595 · CUDA 13.2)처럼 새 세대 GPU 에서는 탐지 이미지(CUDA 11.8·cuDNN 8·TF 2.14,
# 빌드 대상 sm_35~75 + compute_80 PTX)가 실행에 실패할 가능성이 커서 이 점검으로 미리 확인한다.
. "$(dirname "$0")/common.sh"; load_env
QUICK=0
for arg in "$@"; do case "$arg" in --quick) QUICK=1;; -h|--help) sed -n '2,8p' "$0"; exit 0;; *) fail "알 수 없는 인자: $arg";; esac; done
start_log gpu_check
VERDICT="cpu"; REASON=""
step() { printf '\033[1;34m[gpu]\033[0m %s\n' "$*"; }

step "1/5 호스트 NVIDIA 드라이버 (nvidia-smi)"
if command -v nvidia-smi >/dev/null 2>&1 && host=$(nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv,noheader 2>&1); then
  ok "GPU: $host"
  HOST_GPU=1
else
  warn "호스트에서 nvidia-smi 가 없거나 실패 — GPU 없음 또는 드라이버 문제 (${host:-not found})"
  HOST_GPU=0; REASON="호스트 드라이버/nvidia-smi 없음"
fi

step "2/5 nvidia-container-toolkit (docker 가 --gpus 를 쓸 수 있는가)"
if [ "$HOST_GPU" = "1" ]; then
  if command -v nvidia-ctk >/dev/null 2>&1 && docker info 2>/dev/null | grep -qi nvidia; then
    ok "nvidia-ctk $(nvidia-ctk --version 2>/dev/null | head -1 | awk '{print $NF}') · docker 런타임에 nvidia 등록"
    TOOLKIT=1
  else
    warn "nvidia-container-toolkit 이 없거나 docker 에 등록되지 않음 — 설치: apt install nvidia-container-toolkit && sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
    TOOLKIT=0; REASON="nvidia-container-toolkit 미설치/미등록"
  fi
else TOOLKIT=0; fi

step "3/5 탐지 이미지 wbms:0.13 안에서 GPU 가 보이는가 (docker run --gpus device=0 … nvidia-smi)"
if docker image inspect wbms:0.13 >/dev/null 2>&1; then
  IMG=1
  if [ "$TOOLKIT" = "1" ]; then
    if out=$(docker run --rm --gpus device=0 --entrypoint nvidia-smi wbms:0.13 --query-gpu=name,memory.free --format=csv,noheader 2>&1); then
      ok "컨테이너 안 GPU: $out"; CTN_GPU=1
    else
      warn "컨테이너에 GPU 를 넣지 못함: ${out: -200}"; CTN_GPU=0; REASON="docker --gpus 실패"
    fi
  else CTN_GPU=0; fi
else
  warn "wbms:0.13 이미지가 없음 — 실탐지 묶음을 풀고 ./install.sh --with-wbms 뒤 다시 점검 (지금은 캐시된 장면만 탐지 가능)"
  IMG=0; CTN_GPU=0; REASON="${REASON:-탐지 이미지 없음}"
fi

step "4/5 탐지 이미지의 TensorFlow 가 그 GPU 를 쓸 수 있는가"
if [ "${CTN_GPU:-0}" = "1" ]; then
  tfout=$(docker run --rm -i --gpus device=0 --entrypoint python wbms:0.13 - <<'PY' 2>&1
import os, sys
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
import tensorflow as tf
bi = tf.sysconfig.get_build_info()
print("TF", tf.__version__, "CUDA", bi.get("cuda_version"), "cuDNN", bi.get("cudnn_version"), "arch", bi.get("cuda_compute_capabilities"))
gpus = tf.config.list_physical_devices("GPU")
print("GPUS", len(gpus))
if gpus:
    try:
        d = tf.config.experimental.get_device_details(gpus[0]); print("DEVICE", d.get("device_name"), "cc", d.get("compute_capability"))
    except Exception as e:
        print("DEVICE ?", e)
PY
) || true
  echo "$tfout" | grep -E "^(TF|GPUS|DEVICE)" | sed 's/^/    /' || true
  if echo "$tfout" | grep -q "^GPUS [1-9]"; then
    ok "TensorFlow 가 GPU 를 인식"
    TF_GPU=1
  else
    warn "TensorFlow 가 GPU 를 인식하지 못함 (드라이버·CUDA 11.8 호환 문제): $(echo "$tfout" | grep -iE 'error|fail|cuda|cudnn' | tail -2 | tr '\n' ' ' | cut -c1-240)"
    TF_GPU=0; REASON="TF GPU 미인식"
  fi
else TF_GPU=0; fi

step "5/5 GPU 합성곱 연산 실험 (탐지 모델과 같은 종류의 연산, 수 초)"
if [ "$TF_GPU" = "1" ] && [ "$QUICK" = "0" ]; then
  convout=$(timeout 300 docker run --rm -i --gpus device=0 -e TF_FORCE_GPU_ALLOW_GROWTH=true --entrypoint python wbms:0.13 - <<'PY' 2>&1
import os, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
import tensorflow as tf
with tf.device("/GPU:0"):
    x = tf.random.normal([1, 512, 512, 3]); k = tf.random.normal([3, 3, 3, 16])
    t = time.time(); y = tf.nn.conv2d(x, k, 1, "SAME"); y = tf.nn.relu(y); s = float(tf.reduce_sum(y)); dt = time.time() - t
print("CONV_OK", y.shape, round(dt, 3))
PY
) || true
  if echo "$convout" | grep -q "^CONV_OK"; then
    ok "GPU 합성곱 성공: $(echo "$convout" | grep '^CONV_OK')"
    VERDICT="gpu"
  else
    warn "GPU 합성곱 실패 — 이 GPU 세대를 탐지 이미지(CUDA 11.8·cuDNN 8)가 지원하지 않음: $(echo "$convout" | grep -iE 'error|fail|kernel|cudnn|unsupported' | tail -3 | tr '\n' ' ' | cut -c1-300)"
    VERDICT="cpu"; REASON="GPU 커널 실행 실패(이미지 재빌드 필요)"
  fi
elif [ "$TF_GPU" = "1" ]; then
  ok "합성곱 실험 생략(--quick) — TF 가 GPU 를 인식하므로 auto 로 두고 첫 탐지 결과의 gpu_fallback 을 확인"
  VERDICT="auto"
fi

echo
case "$VERDICT" in
  gpu)  ok "판정: 탐지를 GPU 로 돌릴 수 있습니다 → .env 의 WBMS_GPU=auto (기본값) 그대로. 장면당 30~60초." ;;
  auto) ok "판정: GPU 인식됨(연산 실험 생략) → WBMS_GPU=auto. 첫 탐지 잡 결과의 gpu_fallback 유무로 최종 확인." ;;
  cpu)  warn "판정: 탐지는 CPU 로 돕니다 (사유: ${REASON:-GPU 없음}). 장면당 10분 내외. .env 의 WBMS_GPU=auto 그대로 두면 GPU 시도 뒤 자동으로 CPU 재시도하고, 매번 시도하는 시간이 아까우면 WBMS_GPU=cpu 로 고정."
        if [ "${TF_GPU:-0}" = "1" ] || [ "${CTN_GPU:-0}" = "1" ]; then printf '    새 세대 GPU 면 탐지 이미지를 CUDA 12.8 이상·cuDNN 9 베이스로 재빌드해야 GPU 를 씁니다(이미지 제공 기관에 요청). 플랫폼(API·화면·예측)은 영향 없음.\n'; fi ;;
esac
ok "로그 $LOG_FILE"
