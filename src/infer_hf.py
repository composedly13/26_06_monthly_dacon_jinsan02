"""
transformers 기반 추론 스크립트 (vLLM 없이 직접 실행).
vLLM 호환 불가 환경(e.g. Blackwell GPU, 소형 VRAM)에서 사용.

사용 예:
    python src/infer_hf.py \
        --data-csv open/test/test.csv \
        --images-dir open/test \
        --output-path outputs/ \
        --run-name baseline_0.5b_hf
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from transformers import LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor

from utils import parse_answers, normalize_label, extract_json
from prompt import build_baseline_prompt, build_cot_prompt, parse_answer

PROMPT_BUILDERS = {
    "baseline": build_baseline_prompt,
    "cot": build_cot_prompt,
}

MODEL_ID = "llava-hf/llava-onevision-qwen2-0.5b-si-hf"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-csv", required=True)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--prompt-type", choices=list(PROMPT_BUILDERS), default="baseline")
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--output-path", default="./outputs/")
    p.add_argument("--run-name", default=None)
    p.add_argument("--lora-dir", default=None, help="LoRA 체크포인트 경로")
    p.add_argument("--start-from", type=int, default=0, help="이어서 추론할 시작 행 인덱스")
    return p.parse_args()


def load_model(model_id: str, lora_dir: str = None):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        cap = torch.cuda.get_device_capability()
        dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
    processor = LlavaOnevisionProcessor.from_pretrained(model_id)
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if use_cuda else None,
    )
    if lora_dir and Path(lora_dir).exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_dir)
        print(f"LoRA loaded: {lora_dir}")
    if not use_cuda:
        model = model.to("cpu")
    model.eval()
    return model, processor


def load_image(path: Path, img_size: int) -> Image.Image | None:
    try:
        img = Image.open(str(path)).convert("RGB")
        w_pct = img_size / float(img.size[0])
        new_h = int(img.size[1] * w_pct)
        return img.resize((img_size, new_h), Image.LANCZOS)
    except Exception as e:
        print(f"[load_image] {path}: {e}")
        return None


def run_one(model, processor, image: Image.Image, prompt_text: str, max_new_tokens: int) -> str:
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=text, return_tensors="pt").to(model.device, model.dtype)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated = out[0][inputs["input_ids"].shape[-1]:]
    return processor.decode(generated, skip_special_tokens=True)


def _save(df: pd.DataFrame, out_csv: str, detail_csv: str):
    cols = ["sample_id", "label"]
    df[[c for c in cols if c in df.columns]].to_csv(out_csv, index=False)
    df.to_csv(detail_csv, index=False)


def main():
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    run_name = args.run_name or f"{args.prompt_type}_hf_{int(time.time())}"
    out_csv = os.path.join(args.output_path, f"{run_name}_submission.csv")
    detail_csv = os.path.join(args.output_path, f"{run_name}_detail.csv")

    df = pd.read_csv(args.data_csv)
    if args.max_samples:
        df = df.head(args.max_samples).copy()

    df["model_output"] = None
    df["label"] = None

    df["model_output"] = None
    df["label"] = None

    build_prompt_text = PROMPT_BUILDERS[args.prompt_type]

    print(f"Loading model: {args.model_id}")
    model, processor = load_model(args.model_id, lora_dir=args.lora_dir)
    print(f"Model loaded. Device: {next(model.parameters()).device}")

    save_every = 50

    # 이어서 추론: start_from 이전 결과를 기존 detail CSV에서 복원
    if args.start_from > 0 and os.path.exists(detail_csv):
        prev = pd.read_csv(detail_csv)
        for col in ["model_output", "label"]:
            if col in prev.columns:
                df.loc[:args.start_from - 1, col] = prev.loc[:args.start_from - 1, col].values
        print(f"이어서 추론: row {args.start_from}부터 (이전 {args.start_from}개 복원)")

    with tqdm(total=len(df), desc="Inference", unit="sample", initial=args.start_from) as pbar:
        for row_idx, row in df.iloc[args.start_from:].iterrows():
            image_path = Path(args.images_dir) / str(row["image_path"])
            img = load_image(image_path, args.img_size)

            if img is None:
                df.at[row_idx, "label"] = "0"
                df.at[row_idx, "model_output"] = ""
                pbar.update(1)
                continue

            answers = parse_answers(row["answers"])
            prompt_text = build_prompt_text(
                str(row.get("context", "")),
                str(row.get("question", "")),
                answers,
            )

            try:
                text = run_one(model, processor, img, prompt_text, args.max_new_tokens)
                df.at[row_idx, "model_output"] = text
            except Exception as e:
                print(f"[row {row_idx}] run_one error: {e}")
                df.at[row_idx, "label"] = "0"
                df.at[row_idx, "model_output"] = ""
                pbar.update(1)
                continue
            try:
                answers_list = parse_answers(row["answers"])
                label = parse_answer(text, answers_list)
                df.at[row_idx, "label"] = str(label)
            except Exception as e:
                print(f"[row {row_idx}] parse error: {e} | output={text[:80]}")
                df.at[row_idx, "label"] = normalize_label(text)

            if (pbar.n + 1) % save_every == 0 or row_idx == df.index[-1]:
                _save(df, out_csv, detail_csv)

            pbar.update(1)

    _save(df, out_csv, detail_csv)
    print(f"Saved: {out_csv}")
    print(f"Detail: {detail_csv}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"total_elapsed={time.time() - t0:.1f}s")
