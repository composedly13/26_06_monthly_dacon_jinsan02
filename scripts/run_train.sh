#!/bin/bash
# LoRA 학습 실행 스크립트
# 사용: bash scripts/run_train.sh [--use-4bit]
# RTX 5060 8GB: batch-size 1 권장
# RTX A6000 48GB: batch-size 4 + --use-4bit 가능

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CUDA_VISIBLE_DEVICES=0

cd "$REPO_ROOT/src"

python train_lora.py \
  --train-csv  "$REPO_ROOT/data/train100/train100.csv" \
  --images-dir "$REPO_ROOT/data/train100/images" \
  --output-dir "$REPO_ROOT/outputs/lora_checkpoint" \
  --epochs 3 \
  --batch-size 1 \
  --lr 2e-4 \
  "$@"
