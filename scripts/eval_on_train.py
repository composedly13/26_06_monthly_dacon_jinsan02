"""
학습 데이터(정답 있음)로 현재 모델 정확도 측정 + 실패 패턴 분석.
python scripts/eval_on_train.py --n 200
"""
import argparse, sys, time
from pathlib import Path

import torch
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from prompt import parse_answer, build_messages, SYSTEM_PROMPT
from utils import parse_answers

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n",          type=int, default=200)
    p.add_argument("--train-csv",  default="train/train200.csv")
    p.add_argument("--images-dir", default="train/images")
    p.add_argument("--model-id",   default="Qwen/Qwen2.5-VL-3B-Instruct")
    return p.parse_args()

def main():
    args = parse_args()
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from qwen_vl_utils import process_vision_info

    df = pd.read_csv(args.train_csv).head(args.n)
    print(f"평가 샘플: {len(df)}개")

    use_cuda = torch.cuda.is_available()
    cap = torch.cuda.get_device_capability() if use_cuda else (0, 0)
    dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
    device = "cuda" if use_cuda else "cpu"

    cache = Path.home() / ".cache/huggingface/hub" / f"models--{args.model_id.replace('/', '--')}"
    load_path = str(cache) if (cache / "config.json").exists() else args.model_id

    processor = AutoProcessor.from_pretrained(load_path)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        load_path, dtype=dtype, device_map=device, attn_implementation="sdpa"
    ).eval()
    print("모델 로드 완료")

    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    results = []

    with torch.inference_mode():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Eval"):
            opts = parse_answers(row["answers"])
            gt   = int(row["label"])
            img_path = Path(args.images_dir) / Path(row["image_path"]).name
            img = Image.open(str(img_path)).convert("RGB") if img_path.exists() else None
            msgs = build_messages(img, row["context"], row["question"], opts,
                                  include_image=img is not None)
            text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            try:
                img_in, vid_in = process_vision_info([msgs])
                inputs = processor(text=[text], images=img_in, videos=vid_in,
                                   padding=True, return_tensors="pt").to(device)
            except Exception:
                inputs = processor(text=[text], padding=True, return_tensors="pt").to(device)
            out = model.generate(**inputs, max_new_tokens=200,
                                 do_sample=False, pad_token_id=pad_id)
            dec = processor.decode(out[0][inputs["input_ids"].shape[1]:],
                                   skip_special_tokens=True)
            pred = parse_answer(dec, opts)
            results.append({"gt": gt, "pred": pred, "correct": gt == pred,
                            "opts": opts, "question": row["question"],
                            "context": row["context"][:80], "raw": dec[:300]})

    res = pd.DataFrame(results)
    acc = res["correct"].mean()
    print(f"\n=== 결과 ===")
    print(f"정확도: {acc:.4f} ({res['correct'].sum()}/{len(res)})")
    print(f"\n혼동 행렬 (pred↓ gt→):")
    print(pd.crosstab(res["pred"], res["gt"], rownames=["pred"], colnames=["gt"]))

    # 실패 케이스 분석
    wrong = res[~res["correct"]]
    print(f"\n=== 틀린 케이스 분석 ({len(wrong)}개) ===")
    print("gt→pred 패턴:")
    print(wrong.groupby(["gt", "pred"]).size().reset_index(name="count").to_string())
    print()
    print("틀린 케이스 예시 (5개):")
    for _, r in wrong.head(5).iterrows():
        print(f"  gt={r['gt']} pred={r['pred']} | Q: {r['question'][:60]}")
        print(f"  raw: {r['raw'][:200]}")
        print()

    res.to_csv("outputs/eval_train_result.csv", index=False)
    print("저장: outputs/eval_train_result.csv")

if __name__ == "__main__":
    main()
