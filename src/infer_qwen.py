"""
Qwen2.5-VL 추론 스크립트 — 공유 코드(0.99517) 구조 기반.
로컬(8GB): Qwen2.5-VL-3B-Instruct
제출(A6000 48GB): Qwen/Qwen3.5-9B or Qwen2.5-VL-7B-Instruct

사용:
    # 로컬 (3B)
    python src/infer_qwen.py \
        --model-id Qwen/Qwen2.5-VL-3B-Instruct \
        --data-csv open/test/test.csv \
        --images-dir open/test \
        --run-name qwen3b_v1

    # 제출용 (A6000)
    python src/infer_qwen.py \
        --model-id Qwen/Qwen2.5-VL-7B-Instruct \
        --batch-size 16 \
        --run-name qwen7b_submit
"""
import argparse, json, os, time
from pathlib import Path

import torch
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from prompt import parse_answer, build_messages, SYSTEM_PROMPT
from utils import parse_answers


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id",    default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--data-csv",    required=True)
    p.add_argument("--images-dir",  required=True)
    p.add_argument("--output-path", default="./outputs/")
    p.add_argument("--run-name",    default=None)
    p.add_argument("--batch-size",  type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-pixels",  type=int, default=200704)   # ~448×448
    p.add_argument("--min-pixels",  type=int, default=50176)    # ~224×224
    p.add_argument("--no-image",    action="store_true")
    p.add_argument("--use-4bit",    action="store_true",
                   help="4-bit 양자화 (T4 등 VRAM ≤16GB 환경, bitsandbytes 필요)")
    p.add_argument("--start-from",  type=int, default=0,
                   help="이어서 추론할 시작 행 인덱스")
    return p.parse_args()


def load_model(model_id: str, max_pixels: int, min_pixels: int, use_4bit: bool = False):
    use_cuda = torch.cuda.is_available()
    cap = torch.cuda.get_device_capability() if use_cuda else (0, 0)
    dtype = torch.bfloat16 if cap[0] >= 8 else (torch.float16 if use_cuda else torch.float32)
    device = "cuda" if use_cuda else "cpu"

    # 로컬 flat 캐시 경로 우선, 없으면 HF hub에서 다운
    cache_dir = Path.home() / ".cache/huggingface/hub" / f"models--{model_id.replace('/', '--')}"
    load_path = str(cache_dir) if (cache_dir / "config.json").exists() else model_id
    print(f"Loading processor from: {load_path}")

    processor = AutoProcessor.from_pretrained(load_path)
    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        tok.padding_side = "left"
    ip = getattr(processor, "image_processor", None)
    if ip is not None:
        ip.max_pixels = max_pixels
        ip.min_pixels = min_pixels

    model_kwargs = dict(device_map="auto" if use_cuda else None, attn_implementation="sdpa")
    if use_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        print(f"Loading model (4-bit NF4, compute={dtype}) ...")
    else:
        model_kwargs["dtype"] = dtype
        print(f"Loading model ({dtype}) ...")

    model = AutoModelForImageTextToText.from_pretrained(load_path, **model_kwargs).eval()
    print(f"Model loaded on {device}.")
    return model, processor


def prepare_batch(rows, processor, images_dir: str, no_image: bool):
    """배치 입력 준비 (Qwen VL 형식)."""
    texts, all_msgs = [], []
    for r in rows:
        opts = parse_answers(r["answers"])
        img = None
        if not no_image:
            p = Path(images_dir) / Path(r["image_path"]).name
            try:
                img = Image.open(str(p)).convert("RGB")
            except Exception:
                img = None
        msgs = build_messages(img, r["context"], r["question"], opts,
                               include_image=(not no_image and img is not None))
        all_msgs.append(msgs)
        texts.append(processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True))

    # Qwen VL: process_vision_info로 이미지 추출
    if no_image:
        return processor(text=texts, padding=True, return_tensors="pt")
    try:
        from qwen_vl_utils import process_vision_info
        img_in, vid_in = process_vision_info(all_msgs)
        return processor(text=texts, images=img_in, videos=vid_in,
                         padding=True, return_tensors="pt")
    except ImportError:
        # qwen_vl_utils 없으면 이미지 직접 추출
        images = [m[1]["content"][0]["image"] for m in all_msgs
                  if m[1]["content"][0].get("type") == "image"]
        return processor(text=texts, images=images or None,
                         padding=True, return_tensors="pt")


def main():
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    run_name = args.run_name or f"qwen_{int(time.time())}"
    out_csv    = os.path.join(args.output_path, f"{run_name}_submission.csv")
    detail_csv = os.path.join(args.output_path, f"{run_name}_detail.csv")

    df = pd.read_csv(args.data_csv)
    if args.max_samples:
        df = df.head(args.max_samples).copy()
    print(f"{len(df)} rows | batch={args.batch_size} | model={args.model_id} | 4bit={args.use_4bit}")

    model, processor = load_model(args.model_id, args.max_pixels, args.min_pixels, args.use_4bit)
    tok = getattr(processor, "tokenizer", None)
    pad_id = (tok.pad_token_id if tok and tok.pad_token_id is not None
              else (tok.eos_token_id if tok else None))

    gen_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=False)
    if pad_id is not None:
        gen_kwargs["pad_token_id"] = pad_id

    preds, raws = [], []
    rows = df.to_dict("records")
    device = next(model.parameters()).device
    t0 = time.time()

    # 이어서 추론: start_from 이전 결과 복원
    if args.start_from > 0 and os.path.exists(detail_csv):
        prev = pd.read_csv(detail_csv)
        n_restore = min(args.start_from, len(prev))
        for i in range(n_restore):
            lbl = prev.iloc[i].get("label", None)
            raw = prev.iloc[i].get("_raw", "")
            preds.append(int(lbl) if pd.notna(lbl) else 0)
            raws.append(str(raw) if pd.notna(raw) else "")
        rows = rows[n_restore:]
        print(f"이어서 추론: {n_restore}개 복원, {len(rows)}개 남음")

    with torch.inference_mode():
        for s in tqdm(range(0, len(rows), args.batch_size), desc="Inference", unit="batch"):
            batch = rows[s : s + args.batch_size]
            try:
                inputs = prepare_batch(batch, processor, args.images_dir, args.no_image)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                out = model.generate(**inputs, **gen_kwargs)
                trimmed = out[:, inputs["input_ids"].shape[1]:]
                dec = processor.batch_decode(trimmed, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)
                for r, o in zip(batch, dec):
                    opts = parse_answers(r["answers"])
                    preds.append(parse_answer(o, opts))
                    raws.append(o.strip().replace("\n", " ")[:200])
            except Exception as e:
                print(f"[batch {s}] error: {e}")
                for r in batch:
                    opts = parse_answers(r["answers"])
                    preds.append(parse_answer("", opts))
                    raws.append("")

    dt = time.time() - t0
    print(f"Done: {len(preds)} samples in {dt/60:.1f} min ({dt/len(preds)*1000:.0f} ms/sample)")

    df["label"] = preds
    df["_raw"]  = raws
    df[["sample_id", "label"]].to_csv(out_csv, index=False)
    df[["sample_id", "label", "_raw"]].to_csv(detail_csv, index=False)
    print(f"Submission → {out_csv}")

    dist = df["label"].value_counts().sort_index().to_dict()
    print(f"Label dist: {dist}")


if __name__ == "__main__":
    main()
