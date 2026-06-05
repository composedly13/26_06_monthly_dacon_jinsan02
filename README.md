# 2026 성균관대학교 멀티모달 AI 챌린지

[DACON 대회 링크](https://dacon.io/competitions/official/236722/overview/description)

이미지 + 컨텍스트 + 질문이 주어졌을 때, 3개 선택지(0/1/2) 중 가장 적절한 답을 예측하는 멀티모달 QA 태스크입니다.  
사회적 편향(성별·인종 등)이 포함된 BBQ-style 질문에서 근거 없는 답변 대신 "알 수 없음" 선택지를 올바르게 고르는 능력을 평가합니다.

---

## 제출 기록

| 버전 | 모델 | 프롬프트 | 점수 | 비고 |
|------|------|---------|------|------|
| v1 | Qwen2.5-VL-3B-Instruct | v2 ([Visual]/[Textual]/[Logic]) | **0.8063** | 현재 베스트 |
| v3b | Qwen2.5-VL-3B-Instruct | v3 (Unknown-First Rule) | 0.7750 | Unknown 과잉 선택으로 하락 |

> 현재 순위: 약 33위 / 206팀 (2026-06-05 기준)

---

## 로컬 실행 환경

| 항목 | 버전 |
|------|------|
| OS | Windows 11 + WSL2 (Ubuntu 22.04) |
| Python | 3.11 |
| CUDA | 12.x |
| PyTorch | 2.6.0 |
| GPU | RTX 5060 8 GB (로컬) |
| 제출용 GPU | A100 40 GB / A6000 48 GB (클라우드) |

---

## 디렉토리 구조

```
.
├── open/                         # 대회 원본 데이터 (gitignore)
│   ├── train/images/
│   ├── train/train.csv
│   ├── test/images/              # 8,500장
│   └── test/test.csv
│
├── src/
│   ├── infer_qwen.py             # 메인 추론 스크립트 (Qwen2.5-VL / Qwen3.5)
│   ├── infer_hf.py               # HuggingFace 범용 추론 (텍스트 전용)
│   ├── prompt.py                 # 프롬프트 템플릿 v2 + parse_answer()
│   ├── train_lora.py             # LoRA 파인튜닝 스크립트
│   └── utils.py                  # parse_answers() 등 유틸리티
│
├── scripts/
│   ├── eval_on_train.py          # 학습 데이터로 정확도 평가
│   ├── analyze_detail.py         # 추론 결과 상세 분석
│   ├── prepare_train_data.py     # SB-Bench 학습 데이터 준비
│   └── download_sbbench.py       # SB-Bench 데이터셋 다운로드
│
├── notebooks/
│   └── colab_7b_infer.ipynb      # Colab T4에서 7B 4-bit 추론
│
├── submissions/
│   └── qwen3b_v1_full_submission.csv   # 베스트 제출본 (0.8063)
│
└── outputs/                      # 추론 결과 (gitignore)
```

---

## 빠른 시작

### 1. 환경 설정 (WSL / Linux)

```bash
python -m venv ~/dacon-venv
source ~/dacon-venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate qwen-vl-utils pillow pandas tqdm
```

### 2. 추론 실행 (로컬 3B)

```bash
python src/infer_qwen.py \
    --model-id Qwen/Qwen2.5-VL-3B-Instruct \
    --data-csv open/test/test.csv \
    --images-dir open/test \
    --run-name qwen3b_v1
```

완료 후 생성 파일:

| 파일 | 설명 |
|------|------|
| `outputs/{run_name}_submission.csv` | 제출용 (`sample_id`, `label`) |
| `outputs/{run_name}_detail.csv` | 분석용 (모델 원문 출력 포함) |

### 3. 클라우드 추론 (A100, 7B 이상)

```bash
# Qwen2.5-VL-7B (A100 40GB, bfloat16)
python src/infer_qwen.py \
    --model-id Qwen/Qwen2.5-VL-7B-Instruct \
    --batch-size 8 \
    --run-name qwen7b_submit

# Qwen3.5-9B (A100 40GB, bfloat16)
python src/infer_qwen.py \
    --model-id Qwen/Qwen3.5-9B \
    --batch-size 8 \
    --run-name qwen35_9b_submit
```

### 4. 이어서 추론 (세션 끊김 대비)

```bash
python src/infer_qwen.py \
    --model-id Qwen/Qwen2.5-VL-3B-Instruct \
    --data-csv open/test/test.csv \
    --images-dir open/test \
    --run-name qwen3b_v1 \
    --start-from 4000        # 이전 detail CSV에서 복원
```

---

## 프롬프트 설계 (v2)

`src/prompt.py`의 `SYSTEM_PROMPT`는 3단계 구조화 추론을 강제합니다.

```
[Visual]  <이미지에서 보이는 시각적 증거>
[Textual] <컨텍스트에서 명시된 텍스트 증거>
[Logic]   <소거법 또는 직접 매칭>
Answer: <0, 1, 또는 2>
```

**핵심 원칙:**
- 명시적 사실과 시각적 근거만 사용 (고정관념·추측 금지)
- 증거가 불충분하면 반드시 Unknown 옵션 선택
- `parse_answer()`: 5단계 폴백 파싱 (Answer: X → 마지막 숫자 → 옵션 텍스트 → Unknown)

**실험 결과 교훈:**
- Unknown 비율: ~34~35%가 최적 (39% 이상이면 점수 하락)
- "Unknown 먼저" 전략(v3)은 DACON 테스트 분포와 맞지 않아 역효과

---

## 다음 시도 목록

- [ ] Qwen3.5-9B 클라우드(A100) 추론
- [ ] Self-Consistency K=3 (같은 모델 3회 투표)
- [ ] Qwen2.5-VL-7B LoRA 파인튜닝 (RunPod)

---

## 대회 규칙 요약

- 외부 API(OpenAI, Gemini, HuggingFace Inference 등) **사용 불가**
- 2026-06-01 이전 공개된 오픈소스 모델만 허용
- 단순 룰 기반 출력 **불가** — LLM이 직접 생성한 답변이어야 함
- 일 최대 제출 **5회**
