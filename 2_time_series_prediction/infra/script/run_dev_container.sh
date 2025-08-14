#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

docker run -it --rm \
  --name conv-lstm-dev \
  --gpus all \
  -v "$PROJECT_ROOT":/workspace \
  conv-lstm:gpu \
  /bin/bash