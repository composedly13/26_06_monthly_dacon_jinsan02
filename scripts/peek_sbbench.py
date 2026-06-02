"""SB-Bench(ucf-crcv/SB-Bench) 구조/이미지 형식 확인 — streaming으로 가볍게 peek."""
import datasets
from huggingface_hub import HfApi

print("datasets", datasets.__version__)

# 1) 레포 파일/용량 확인
api = HfApi()
try:
    info = api.dataset_info("ucf-crcv/SB-Bench", files_metadata=True)
    total = 0
    splits = set()
    for s in (info.siblings or []):
        sz = s.size or 0
        total += sz
        splits.add(s.rfilename.split("/")[0])
    print(f"repo total size ~ {total/1e6:.1f} MB, top-level entries: {sorted(splits)[:12]}")
except Exception as e:
    print("dataset_info error:", e)

# 2) streaming으로 첫 행 구조 확인
try:
    ds = datasets.load_dataset("ucf-crcv/SB-Bench", split="train", streaming=True)
    for i, row in enumerate(ds):
        keys = list(row.keys())
        print("FIELDS:", keys)
        for k in keys:
            v = row[k]
            t = type(v).__name__
            if hasattr(v, "size"):  # PIL image
                print(f"  {k}: PIL image size={v.size}")
            else:
                s = str(v)
                print(f"  {k} ({t}): {s[:120]}")
        break
except Exception as e:
    print("streaming error:", e)
