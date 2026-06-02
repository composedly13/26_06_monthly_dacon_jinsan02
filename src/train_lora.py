"""
LLaVA-OneVision 0.5B LoRA 파인튜닝 스크립트.
RTX 5060 8GB / RTX A6000 48GB 모두 지원.

사용 예:
    python src/train_lora.py \
        --train-csv data/train100/train100.csv \
        --images-dir data/train100/images \
        --output-dir outputs/lora_checkpoint \
        --epochs 3 \
        --batch-size 2
"""
import argparse
import json
import os
from pathlib import Path

import torch
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from transformers import (
    LlavaOnevisionForConditionalGeneration,
    LlavaOnevisionProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# ── 상수 ──────────────────────────────────────────────────────────────────────
MODEL_ID = "llava-hf/llava-onevision-qwen2-0.5b-si-hf"
IMG_SIZE = 336

ANSWER_TEMPLATE = '{{"reason": "Based on the given information.", "answer_id": "{aid}"}}'

# ── 데이터셋 ───────────────────────────────────────────────────────────────────
class BiasVQADataset(Dataset):
    def __init__(self, csv_path: str, images_dir: str, processor, img_size: int = IMG_SIZE):
        self.df = pd.read_csv(csv_path)
        self.images_dir = Path(images_dir)
        self.processor = processor
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.images_dir / str(row["image_path"]).replace("./images/", "")
        try:
            img = Image.open(img_path).convert("RGB")
            w_pct = self.img_size / img.size[0]
            img = img.resize((self.img_size, int(img.size[1] * w_pct)), Image.LANCZOS)
        except Exception:
            img = Image.new("RGB", (self.img_size, self.img_size), (128, 128, 128))

        answers = json.loads(row["answers"])
        label = str(row["label"])
        context = str(row.get("context", ""))
        question = str(row.get("question", ""))

        prompt = (
            f"Context: {context}\n"
            f"Question: {question}\n"
            f"Options:\n0. {answers[0]}\n1. {answers[1]}\n2. {answers[2]}\n"
            "Give the output in strict JSON: "
            '{"reason": "...", "answer_id": "<0,1,2>"}'
        )

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": ANSWER_TEMPLATE.format(aid=label),
            },
        ]

        text = self.processor.apply_chat_template(conversation, add_generation_prompt=False)
        return img, text


def collate_fn(batch, processor, device, dtype):
    images, texts = zip(*batch)
    inputs = processor(
        images=list(images),
        text=list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )
    labels = inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    inputs["labels"] = labels
    return {k: v.to(device, dtype if v.is_floating_point() else None)
            for k, v in inputs.items()}


# ── 모델 로드 ─────────────────────────────────────────────────────────────────
def load_model_with_lora(model_id: str, use_4bit: bool = False):
    # Blackwell(SM 12.0) / RTX 50xx: bfloat16 네이티브 지원, fp16 cublas 불안정
    # bitsandbytes 4-bit도 SM 12.0에서 cublas 오류 발생 → bf16 직접 사용
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        # SM >= 8.0 (Ampere+) 은 bf16 지원; Blackwell은 SM 12.0
        dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
    else:
        dtype = torch.float32

    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    # gradient checkpointing: 8GB VRAM에서 max_length=2048 안정 실행
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ── 학습 루프 ─────────────────────────────────────────────────────────────────
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
    print(f"Device: {device}  dtype: {dtype}  compute_cap: {torch.cuda.get_device_capability() if device == 'cuda' else 'N/A'}")

    processor = LlavaOnevisionProcessor.from_pretrained(MODEL_ID)
    model = load_model_with_lora(MODEL_ID, use_4bit=args.use_4bit)
    if device == "cpu":
        model = model.to(device)

    dataset = BiasVQADataset(args.train_csv, args.images_dir, processor)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, processor, device, dtype),
        num_workers=0,
    )

    total_steps = (len(loader) // args.grad_accum) * args.epochs
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))

    os.makedirs(args.output_dir, exist_ok=True)
    best_loss = float("inf")
    print(f"Total steps: {total_steps}  (grad_accum={args.grad_accum}, eff_batch={args.batch_size * args.grad_accum})")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        with tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}") as pbar:
            for step, batch in enumerate(pbar):
                outputs = model(**batch)
                loss = outputs.loss / args.grad_accum
                loss.backward()
                total_loss += loss.item() * args.grad_accum

                if (step + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                pbar.set_postfix(loss=f"{(loss.item() * args.grad_accum):.4f}")

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}: avg_loss={avg_loss:.4f}")

        ckpt_path = os.path.join(args.output_dir, f"epoch{epoch}")
        model.save_pretrained(ckpt_path)
        processor.save_pretrained(ckpt_path)
        print(f"Checkpoint saved → {ckpt_path}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(args.output_dir, "best")
            model.save_pretrained(best_path)
            processor.save_pretrained(best_path)
            print(f"Best checkpoint updated → {best_path}")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Best checkpoint: {os.path.join(args.output_dir, 'best')}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", required=True)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--output-dir", default="./outputs/lora_checkpoint")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4,
                   help="gradient accumulation steps (실효 배치 = batch_size × grad_accum)")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--use-4bit", action="store_true",
                   help="4-bit QLoRA (VRAM 절약, A6000/3090 권장)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
