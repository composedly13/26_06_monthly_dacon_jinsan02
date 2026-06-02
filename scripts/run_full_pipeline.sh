#!/usr/bin/env bash
# 전체 파이프라인: 데이터 준비 → 학습 → 체크 (다운로드 완료 후 실행)
set -e

N=${1:-1000}
EPOCHS=${2:-5}
GRAD_ACCUM=4
LR=2e-4

echo "=== 파이프라인 시작: N=${N}, EPOCHS=${EPOCHS}, grad_accum=${GRAD_ACCUM} ==="

cd /mnt/c/dacon
source ~/dacon-venv/bin/activate

# 1. 학습 데이터 준비
echo "[1/3] 데이터 준비 (${N}개)..."
python scripts/prepare_train_data.py \
    --sb-dir /home/jinsan/sb_bench_data \
    --out-dir train \
    --n "${N}"

# 2. LoRA 학습
echo "[2/3] LoRA 학습 시작..."
python src/train_lora.py \
    --train-csv "train/train${N}.csv" \
    --images-dir train/images \
    --output-dir outputs/lora_checkpoint \
    --epochs "${EPOCHS}" \
    --batch-size 1 \
    --grad-accum "${GRAD_ACCUM}" \
    --lr "${LR}"

# 3. 제출 검증
echo "[3/3] 제출 검증..."
python scripts/check_submission.py \
    --lora-dir outputs/lora_checkpoint/best \
    --skip-inference

echo "=== 파이프라인 완료 ==="
