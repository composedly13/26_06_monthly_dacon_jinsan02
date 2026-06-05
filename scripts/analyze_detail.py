import pandas as pd
import re

df = pd.read_csv("outputs/qwen3b_v1_full_detail.csv")

print("=== 기본 통계 ===")
print(f"총 샘플: {len(df)}")
print(f"라벨 분포: {df['label'].value_counts().sort_index().to_dict()}")
print()

# 빈 응답
no_raw = df[df['_raw'].str.strip() == '']
print(f"빈 응답(에러): {len(no_raw)}개")

# Answer: 패턴 있는지
has_pat = df['_raw'].str.contains(r'Answer\s*[:\-]', case=False, na=False)
print(f"Answer: 패턴 있음: {has_pat.sum()}개")
print(f"Answer: 패턴 없음: {(~has_pat & (df['_raw'].str.strip() != '')).sum()}개")
print()

# [Visual]/[Textual]/[Logic] 포맷 준수 여부
has_visual = df['_raw'].str.contains(r'\[Visual\]', na=False)
has_logic  = df['_raw'].str.contains(r'\[Logic\]', na=False)
print(f"[Visual] 포함: {has_visual.sum()}개 ({has_visual.sum()/len(df)*100:.1f}%)")
print(f"[Logic] 포함:  {has_logic.sum()}개 ({has_logic.sum()/len(df)*100:.1f}%)")
print()

# unknown 폴백으로 처리된 케이스 추정
# parse_answer fallback: Answer 패턴 없고 숫자도 없는 경우
no_digit = df['_raw'].str.extract(r'\b([012])\b')[0].isna()
fallback  = (~has_pat) & no_digit & (df['_raw'].str.strip() != '')
print(f"폴백 처리(숫자 없음): {fallback.sum()}개")
print()

# 라벨 2로 편향된 케이스
print("=== 라벨별 평균 raw 길이 ===")
for lbl in [0, 1, 2]:
    sub = df[df['label'] == lbl]
    avg_len = sub['_raw'].str.len().mean()
    print(f"  라벨 {lbl}: {len(sub)}개, 평균 출력길이 {avg_len:.0f}자")
print()

# Answer 패턴 없는 샘플 예시
no_pat_df = df[~has_pat & (df['_raw'].str.strip() != '')]
print("=== Answer 패턴 없는 샘플 예시 (5개) ===")
for i, row in no_pat_df.head(5).iterrows():
    print(f"[idx {i}] label={row['label']}")
    print(f"  raw: {row['_raw'][:200]}")
    print()

# 응답이 잘린 경우 (200자 제한)
truncated = df['_raw'].str.len() >= 198
print(f"응답 잘림 의심 (198자+): {truncated.sum()}개")
