"""
SB-Bench 미완료 parquet 다운로드 (파일별, 재시도 포함).
real 7개는 이미 완료. synthetic 누락분만 받음.
"""
import os, time
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ID   = "ucf-crcv/SB-Bench"
LOCAL_DIR = Path("/home/jinsan/sb_bench_data")   # hf_hub_download가 data/ 서브폴더 포함 저장
TOKEN_FILE = os.path.expanduser("~/.cache/huggingface/token")
TOKEN = open(TOKEN_FILE).read().strip() if os.path.exists(TOKEN_FILE) else None

LOCAL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = LOCAL_DIR / "data"   # 실제 parquet 저장 위치
DATA_DIR.mkdir(exist_ok=True)

# 전체 파일 목록 (HF repo 내 경로: data/xxx.parquet)
REAL_FILES = [f"data/real-{i:05d}-of-00007.parquet" for i in range(7)]
SYNTH_FILES = [f"data/synthetic-{i:05d}-of-00020.parquet" for i in range(20)]
ALL_FILES = REAL_FILES + SYNTH_FILES

done, skipped, failed = 0, 0, []

print(f"대상 parquet: {len(ALL_FILES)}개 (real 7 + synthetic 20)")
print(f"저장 경로: {DATA_DIR}")
print()

for fname in ALL_FILES:
    # hf_hub_download(local_dir=LOCAL_DIR, filename="data/xxx.parquet")
    # → 저장 위치: LOCAL_DIR/data/xxx.parquet
    dest = DATA_DIR / Path(fname).name
    if dest.exists() and dest.stat().st_size > 1024 * 100:
        skipped += 1
        print(f"  [SKIP] {Path(fname).name} ({dest.stat().st_size // 1024 // 1024}MB)")
        continue

    print(f"  [DOWN] {Path(fname).name} ...", flush=True)
    for attempt in range(3):
        try:
            hf_hub_download(
                repo_id=REPO_ID,
                filename=fname,
                repo_type="dataset",
                token=TOKEN,
                local_dir=str(LOCAL_DIR),
            )
            sz = dest.stat().st_size // 1024 // 1024
            print(f"  ✅ {Path(fname).name} ({sz}MB)", flush=True)
            done += 1
            break
        except Exception as e:
            print(f"  ⚠️  attempt {attempt+1}/3: {e}", flush=True)
            time.sleep(10)
    else:
        failed.append(Path(fname).name)
        print(f"  ❌ {Path(fname).name} 실패", flush=True)

print(f"\n완료: {done}개 신규, {skipped}개 스킵, {len(failed)}개 실패")
if failed:
    print("실패 목록:", failed)
print("Download complete!")
