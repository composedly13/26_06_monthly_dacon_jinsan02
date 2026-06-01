# 2026 성균관대학교 멀티모달 AI 챌린지

[DACON 대회 링크](https://dacon.io/competitions/official/236722/overview/description)

이미지 + 컨텍스트 + 질문이 주어졌을 때, 3개 선택지(0/1/2) 중 가장 적절한 답을 예측하는 멀티모달 QA 태스크입니다.  
사회적 편향(성별·인종 등)이 포함된 질문에서 근거 없는 답변 대신 "알 수 없음" 선택지를 올바르게 고르는 능력을 평가합니다.

---

## 평가 환경

| 항목 | 버전 |
|------|------|
| OS | Ubuntu 20.04 |
| Python | 3.10 |
| CUDA | 12.4 |
| PyTorch | 2.6.0 |
| GPU | RTX A6000 48 GB |

---

## 디렉토리 구조

```
.
├── open/                       # 대회 원본 데이터 (다운로드 후 배치)
│   ├── train/
│   │   ├── images/
│   │   └── train.csv
│   ├── test/
│   │   ├── images/
│   │   └── test.csv
│   └── sample_submission.csv
│
├── src/
│   ├── infer.py                # 메인 추론 스크립트
│   ├── prompt.py               # 프롬프트 템플릿 (baseline / cot)
│   ├── utils.py                # 이미지 로드, JSON 파싱, 라벨 정규화
│   └── models/
│       └── llava_onevision.py  # vLLM 엔진 초기화
│
├── configs/
│   ├── baseline.yaml
│   └── cot_0.5b.yaml
│
├── scripts/
│   ├── run_baseline.sh         # 베이스라인 실행 스크립트
│   └── run_cot.sh              # CoT 프롬프트 실험
│
├── outputs/                    # 추론 결과 저장 (gitignore)
└── requirements.txt
```

---

## 처음부터 끝까지 실행하기

### 0. 데이터 준비

DACON에서 데이터를 다운로드하여 아래 구조로 배치합니다.

```
open/
├── train/images/   ← 학습 이미지
├── train/train.csv
├── test/images/    ← 테스트 이미지 (8,500장)
├── test/test.csv
└── sample_submission.csv
```

### 1. 환경 설정

```bash
conda create -n dacon_env python=3.10 -y
conda activate dacon_env

# PyTorch 2.6.0 + CUDA 12.4
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# 나머지 의존성
pip install -r requirements.txt
```

### 2. 베이스라인 추론 실행

```bash
bash scripts/run_baseline.sh
```

완료되면 `outputs/` 아래 파일 2개가 생성됩니다.

| 파일 | 설명 |
|------|------|
| `baseline_0.5b_submission.csv` | 제출용 (`sample_id`, `label`) |
| `baseline_0.5b_detail.csv` | 분석용 (모델 원문 출력 포함) |

### 3. 제출

`outputs/baseline_0.5b_submission.csv`를 DACON에 업로드합니다.

---

## 실험 목록

| run_name | 모델 | 프롬프트 | 비고 |
|----------|------|---------|------|
| baseline_0.5b | llava-onevision-qwen2-**0.5b**-si-hf | baseline | 기본 제출용 |
| cot_0.5b | llava-onevision-qwen2-**0.5b**-si-hf | cot | CoT 추론 |

> A6000 48GB 환경에서는 7B 모델(`llava-onevision-qwen2-7b-si-hf`)도 동일 스크립트에 `--model-id` 인자만 교체하여 실행 가능합니다.

---

## 규칙 요약

- 외부 API(OpenAI, Gemini, HuggingFace Inference 등) **사용 불가**
- 2026-06-01 이전 공개된 오픈소스 모델만 허용
- 단순 룰 기반 출력 **불가** — LLM이 직접 생성한 답변이어야 함
- 일 최대 제출 **5회**
