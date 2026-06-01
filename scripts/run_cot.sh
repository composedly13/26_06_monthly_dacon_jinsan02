#!/bin/bash
# CoT 프롬프트 실험 (A6000 48GB 기준 0.5b / 7b 모두 가능)
# 실행: bash scripts/run_cot.sh [선택: --model-id llava-hf/llava-onevision-qwen2-7b-si-hf]

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
  --prompt-type cot \
  --batch-size  8 \
  --img-size    512 \
  --max-tokens  512 \
  --output-path "$REPO_ROOT/outputs" \
  --run-name    "cot_0.5b" \
  "$@"
