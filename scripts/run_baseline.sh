#!/bin/bash
# 베이스라인 추론 실행
# 평가환경: RTX A6000 48GB / Python 3.10 / CUDA 12.4 / PyTorch 2.6.0
# 실행: bash scripts/run_baseline.sh [선택: --max-samples N]

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$REPO_ROOT/open"
SRC_DIR="$REPO_ROOT/src"

export CUDA_VISIBLE_DEVICES=0

cd "$SRC_DIR"

python infer.py \
  --data-csv   "$DATA_DIR/test/test.csv" \
  --images-dir "$DATA_DIR/test" \
  --model-id   "llava-hf/llava-onevision-qwen2-0.5b-si-hf" \
  --prompt-type baseline \
  --batch-size  16 \
  --img-size    512 \
  --max-tokens  256 \
  --output-path "$REPO_ROOT/outputs" \
  --run-name    "baseline_0.5b" \
  "$@"
