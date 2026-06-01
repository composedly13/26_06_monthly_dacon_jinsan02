"""
메인 추론 스크립트.

사용 예:
    python src/infer.py \
        --data-csv open/test/test.csv \
        --images-dir open/test \
        --model-id llava-hf/llava-onevision-qwen2-0.5b-si-hf \
        --prompt-type baseline \
        --batch-size 16 \
        --output-path outputs/
"""
import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from utils import load_image, parse_answers, normalize_label, extract_json
from prompt import build_baseline_prompt, build_cot_prompt
from models.llava_onevision import init_llm, build_prompt, make_sampling_params

PROMPT_BUILDERS = {
    "baseline": build_baseline_prompt,
    "cot": build_cot_prompt,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-csv", required=True)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--model-id", default="llava-hf/llava-onevision-qwen2-0.5b-si-hf")
    p.add_argument("--prompt-type", choices=list(PROMPT_BUILDERS), default="baseline")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-path", default="./outputs/")
    p.add_argument("--run-name", default=None, help="출력 파일 접두어 (없으면 타임스탬프)")
    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    run_name = args.run_name or f"{args.prompt_type}_{int(time.time())}"
    out_csv = os.path.join(args.output_path, f"{run_name}_submission.csv")
    detail_csv = os.path.join(args.output_path, f"{run_name}_detail.csv")

    df = pd.read_csv(args.data_csv)
    if args.max_samples:
        df = df.head(args.max_samples).copy()

    df["model_output"] = None
    df["label"] = None

    build_prompt_text = PROMPT_BUILDERS[args.prompt_type]

    llm = init_llm(args.model_id, seed=args.seed)
    sampling_params = make_sampling_params(args.temperature, args.max_tokens)

    inputs, batch_indices = [], []

    with tqdm(total=len(df), desc="Inference", unit="sample") as pbar:
        for row_idx, row in df.iterrows():
            image_path = Path(args.images_dir) / str(row["image_path"])
            img = load_image(image_path, img_size=args.img_size)

            if img is None:
                df.at[row_idx, "label"] = "0"
                df.at[row_idx, "model_output"] = ""
                _save(df, out_csv, detail_csv)
                pbar.update(1)
                continue

            answers = parse_answers(row["answers"])
            prompt_text = build_prompt_text(
                str(row.get("context", "")),
                str(row.get("question", "")),
                answers,
            )

            inputs.append({
                "prompt": build_prompt(prompt_text),
                "multi_modal_data": {"image": img},
            })
            batch_indices.append(row_idx)

            is_ready = len(inputs) >= args.batch_size
            is_last = row_idx == df.index[-1]

            if is_ready or is_last:
                outputs = llm.generate(inputs, sampling_params, use_tqdm=False)

                for idx, o in zip(batch_indices, outputs):
                    text = o.outputs[0].text
                    df.at[idx, "model_output"] = text
                    try:
                        parsed = extract_json(text)
                        df.at[idx, "label"] = normalize_label(parsed.get("answer_id"))
                    except Exception:
                        df.at[idx, "label"] = "0"

                _save(df, out_csv, detail_csv)
                pbar.update(len(batch_indices))
                inputs, batch_indices = [], []

    print(f"Saved: {out_csv}")
    print(f"Detail: {detail_csv}")


def _save(df: pd.DataFrame, out_csv: str, detail_csv: str):
    cols = ["sample_id", "label"]
    df[[c for c in cols if c in df.columns]].to_csv(out_csv, index=False)
    df.to_csv(detail_csv, index=False)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"total_elapsed={time.time() - t0:.1f}s")
