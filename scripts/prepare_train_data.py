"""
SB-Bench parquet → 대회 포맷 train100 변환 스크립트.
이미지는 parquet 내 바이트로 포함되어 있어 별도 다운로드 불필요.

사용:
    python scripts/prepare_train_data.py \
        --sb-dir /home/jinsan/sb_bench_data \
        --out-dir /mnt/c/dacon/train \
        --n 100
"""
import argparse
import json
import io
import random
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm.auto import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sb-dir", required=True, help="snapshot_download 결과 폴더")
    p.add_argument("--out-dir", default="/mnt/c/dacon/train")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def extract_image(cell) -> Image.Image:
    """HuggingFace Image feature (dict with 'bytes') 또는 raw bytes → PIL Image."""
    if isinstance(cell, dict):
        raw = cell.get("bytes") or cell.get("path")
        if isinstance(raw, bytes):
            return Image.open(io.BytesIO(raw)).convert("RGB")
    if isinstance(cell, bytes):
        return Image.open(io.BytesIO(cell)).convert("RGB")
    raise ValueError(f"Unknown image format: {type(cell)}")


def main():
    args = parse_args()
    random.seed(args.seed)
    sb = Path(args.sb_dir)
    out = Path(args.out_dir)
    img_out = out / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    # parquet 로드
    data_dir = sb / "data" if (sb / "data").exists() else sb
    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files under {data_dir}")

    print(f"Loading {len(parquet_files)} parquet files...")
    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    print(f"Total SB-Bench rows: {len(df)}")

    # 이미지 컬럼 감지 (file_name 또는 image)
    img_col = "file_name" if "file_name" in df.columns else "image"
    print(f"Image column: '{img_col}'")

    # 카테고리 균형 샘플링
    per_cat = max(1, args.n // 9)
    sampled = (
        df.groupby("category", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), per_cat), random_state=args.seed))
    )
    if len(sampled) < args.n:
        remaining = df.drop(sampled.index)
        extra = remaining.sample(
            min(args.n - len(sampled), len(remaining)), random_state=args.seed
        )
        sampled = pd.concat([sampled, extra], ignore_index=True)

    sampled = sampled.head(args.n).reset_index(drop=True)
    print(f"Sampled {len(sampled)} rows")
    print(f"Category dist:\n{sampled['category'].value_counts().to_string()}")

    # 변환 + 이미지 저장
    rows = []
    for i, row in tqdm(sampled.iterrows(), total=len(sampled), desc="Saving images"):
        img_fname = f"train_img_{i:04d}.jpg"
        img_path = img_out / img_fname

        try:
            img = extract_image(row[img_col])
            img.save(img_path, "JPEG", quality=90)
        except Exception as e:
            print(f"  [WARN] row {i} image error: {e} — using gray placeholder")
            Image.new("RGB", (336, 336), (128, 128, 128)).save(img_path)

        answers = json.dumps([str(row["ans0"]), str(row["ans1"]), str(row["ans2"])])
        rows.append({
            "sample_id": f"TRAIN_{i:04d}",
            "image_path": f"./images/{img_fname}",
            "context": str(row.get("context", "")),
            "question": str(row.get("question", "")),
            "answers": answers,
            "label": str(int(row["label"])),
        })

    out_df = pd.DataFrame(rows)
    out_csv = out / f"train{len(out_df)}.csv"
    out_df.to_csv(out_csv, index=False)

    print(f"\n✅ Saved {len(out_df)} rows → {out_csv}")
    print(f"Label distribution:\n{out_df['label'].value_counts().to_string()}")
    print(f"Images → {img_out}")
    print("Done.")


if __name__ == "__main__":
    main()
