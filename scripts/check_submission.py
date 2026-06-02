"""
제출 전 자동 검증 스크립트.
대회 평가환경 기준: Python 3.10, CUDA 12.4, PyTorch 2.6.0, RTX A6000 48GB

사용:
    python scripts/check_submission.py
    python scripts/check_submission.py --lora-dir outputs/lora_checkpoint/best  # LoRA 체크포인트 포함
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch


def check_python():
    v = sys.version_info
    ok = v.major == 3 and v.minor == 10
    print(f"  Python {v.major}.{v.minor}.{v.micro}  {'✅' if ok else '⚠️ 대회환경 3.10'}")
    return ok


def check_torch():
    print(f"  PyTorch {torch.__version__}")
    major, minor = [int(x) for x in torch.__version__.split(".")[:2]]
    if (major, minor) < (2, 6):
        print("  ⚠️  대회환경 PyTorch 2.6.0 이상 권장")
        return False
    print("  ✅ PyTorch 버전 OK")
    return True


def check_cuda():
    if not torch.cuda.is_available():
        print("  ⚠️  CUDA 없음 — CPU 추론으로 동작 (대회환경은 A6000)")
        return False
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  CUDA: {torch.version.cuda}")
    print("  ✅ CUDA OK")
    return True


def check_packages():
    required = ["transformers", "peft", "PIL", "tqdm"]
    all_ok = True
    for pkg in required:
        try:
            m = __import__(pkg)
            ver = getattr(m, "__version__", "?")
            print(f"  ✅ {pkg} {ver}")
        except ImportError:
            print(f"  ❌ {pkg} 없음")
            all_ok = False
    return all_ok


def check_data_files():
    base = Path(__file__).parent.parent / "open"
    files = {
        "test/test.csv": "테스트 데이터",
        "test/images": "테스트 이미지 폴더",
        "sample_submission.csv": "제출 양식",
    }
    all_ok = True
    for rel, name in files.items():
        p = base / rel
        if p.exists():
            if p.is_file():
                size = p.stat().st_size
                print(f"  ✅ {name} ({size/1024:.1f} KB)")
            else:
                n = len(list(p.iterdir()))
                print(f"  ✅ {name} ({n}개 파일)")
        else:
            print(f"  ❌ {name} 없음: {p}")
            all_ok = False
    return all_ok


def check_submission_format(csv_path: Path):
    df = pd.read_csv(csv_path)
    errors = []
    # 필수 컬럼
    for col in ["sample_id", "label"]:
        if col not in df.columns:
            errors.append(f"컬럼 '{col}' 없음")
    # label 범위
    if "label" in df.columns:
        invalid = df[~df["label"].isin([0, 1, 2])]
        if len(invalid):
            errors.append(f"label 범위 오류 {len(invalid)}건 (0/1/2 만 허용)")
    # 샘플 수
    expected = 8500
    if len(df) != expected:
        errors.append(f"샘플 수 {len(df)} (예상 {expected})")
    # sample_id 형식
    if "sample_id" in df.columns:
        bad = df[~df["sample_id"].str.match(r"TEST_\d{4}")]
        if len(bad):
            errors.append(f"sample_id 형식 오류 {len(bad)}건")

    if errors:
        for e in errors:
            print(f"  ❌ {e}")
        return False
    print(f"  ✅ 형식 OK — {len(df)}개 샘플, label 분포: {df['label'].value_counts().to_dict()}")
    return True


def check_inference_speed(lora_dir: str = None, n_samples: int = 5):
    """소량 추론으로 속도 추정."""
    print(f"\n  샘플 {n_samples}개 추론 속도 테스트...")
    try:
        from transformers import LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor
        from PIL import Image

        model_id = "llava-hf/llava-onevision-qwen2-0.5b-si-hf"
        use_cuda = torch.cuda.is_available()
        dtype = torch.bfloat16 if use_cuda else torch.float32

        processor = LlavaOnevisionProcessor.from_pretrained(model_id)
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype,
            device_map="auto" if use_cuda else None,
        )
        if lora_dir and Path(lora_dir).exists():
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, lora_dir)
            print(f"  LoRA 체크포인트 로드: {lora_dir}")
        if not use_cuda:
            model = model.to("cpu")
        model.eval()

        test_csv = Path(__file__).parent.parent / "open/test/test.csv"
        df = pd.read_csv(test_csv).head(n_samples)
        img_base = Path(__file__).parent.parent / "open/test/images"

        results = []
        start_all = time.time()
        for _, row in df.iterrows():
            img_file = img_base / str(row["image_path"]).replace("./images/", "")
            img = Image.open(img_file).convert("RGB") if img_file.exists() \
                  else Image.new("RGB", (336, 336), (128, 128, 128))
            answers = json.loads(row["answers"])
            prompt = (
                f"Context: {row['context']}\nQuestion: {row['question']}\n"
                f"Options:\n0. {answers[0]}\n1. {answers[1]}\n2. {answers[2]}\n"
                "Answer with only the option number (0, 1, or 2)."
            )
            conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            text = processor.apply_chat_template(conv, add_generation_prompt=True)
            inputs = processor(images=[img], text=[text], return_tensors="pt")
            if use_cuda:
                inputs = {k: v.to("cuda", dtype if v.is_floating_point() else None)
                          for k, v in inputs.items()}

            t0 = time.time()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
            elapsed = time.time() - t0
            decoded = processor.decode(out[0], skip_special_tokens=True).strip()
            pred = 0
            for c in decoded[::-1]:
                if c in "012":
                    pred = int(c)
                    break
            results.append((row["sample_id"], pred, elapsed))

        total = time.time() - start_all
        avg = total / n_samples
        est_full = avg * 8500 / 60
        print(f"  ✅ 평균 {avg:.2f}s/샘플 → 8500개 추정 {est_full:.0f}분")
        print(f"  {'✅' if avg <= 0.5 else '⚠️ 대회기준 0.5s 초과'} 속도 기준: 0.5s/샘플")
        print(f"  샘플 결과: {[(r[0], r[1]) for r in results]}")
        return results

    except Exception as e:
        print(f"  ❌ 추론 테스트 실패: {e}")
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lora-dir", default=None, help="LoRA 체크포인트 경로 (선택)")
    p.add_argument("--output-csv", default=None, help="검증할 제출 CSV (선택)")
    p.add_argument("--skip-inference", action="store_true", help="추론 속도 테스트 생략")
    args = p.parse_args()

    print("=" * 55)
    print("  제출 전 자동 검증")
    print("=" * 55)

    results = {}

    print("\n[1] 실행 환경")
    results["python"] = check_python()
    results["torch"] = check_torch()
    results["cuda"] = check_cuda()

    print("\n[2] 필수 패키지")
    results["packages"] = check_packages()

    print("\n[3] 데이터 파일")
    results["data"] = check_data_files()

    if args.output_csv:
        print(f"\n[4] 제출 CSV 형식 검증: {args.output_csv}")
        results["format"] = check_submission_format(Path(args.output_csv))

    if not args.skip_inference:
        print("\n[5] 추론 속도 테스트 (5샘플)")
        check_inference_speed(args.lora_dir)

    print("\n" + "=" * 55)
    critical = ["packages", "data"]
    failed = [k for k in critical if not results.get(k, True)]
    if failed:
        print(f"  ❌ 실패 항목: {failed}")
        sys.exit(1)
    else:
        print("  ✅ 핵심 항목 모두 통과. 제출 준비 완료!")
    print("=" * 55)


if __name__ == "__main__":
    main()
