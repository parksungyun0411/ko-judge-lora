# 한국어 LLM 평가자 LoRA (ko-judge-lora)

[![CI](https://github.com/parksungyun0411/ko-judge-lora/actions/workflows/ci.yml/badge.svg)](https://github.com/parksungyun0411/ko-judge-lora/actions/workflows/ci.yml) ![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white) ![상태](https://img.shields.io/badge/상태-진행_중-lightgrey) [![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

한국어 응답을 1~5점으로 채점하는 소형 LLM 평가자(LLM-as-a-judge)를 사람 점수로 LoRA 학습하고, 사람 점수와 얼마나 맞는지와 평가 대상 모델들의 순위를 얼마나 재현하는지를 잽니다. 같은 평가자를 AWQ 4bit로 양자화한 뒤 vLLM으로 서빙해 양자화 전후의 품질 차이와 100만 건당 비용도 함께 봅니다.

데이터는 한국어 사람 평가 데이터셋 KUDGE([Son et al., 2024](https://arxiv.org/abs/2409.11239))입니다. 31개 모델이 87개 한국어 질문에 쓴 응답 2,506건에 평가자 두 명이 각각 점수를 매겼습니다. 학습용 분할이 없는 평가 전용 데이터라, 질문 단위 교차검증과 평가 대상 모델 보류로 학습·평가를 나눴습니다.

> **진행 중입니다.** 데이터 확인과 프로파일([reports/dataset_profile.md](reports/dataset_profile.md))은 실제 데이터로 끝냈습니다. 학습·서빙이 필요한 단계는 코드와 테스트만 있고 GPU에서 실행하지 않았습니다. 모델 결과 표는 모두 측정 전입니다.

<img src="docs/diagrams/architecture.png" alt="ko-judge-lora 구성: 데이터 → 분할 → 학습 → 서빙 → 평가" width="900">

## 구현 범위

- KUDGE 내려받기: GitHub 고정 커밋에서 받아 sha256 확인, 저장소에는 넣지 않음
- 데이터 확인 기록: 행·질문·모델 수, 점수 분포, 라이선스 근거, 논문 서술과 배포본의 차이([docs/kudge_verification.md](docs/kudge_verification.md))
- 분할: 질문 5-fold와 평가 대상 모델 8개 보류를 한 번의 학습으로 함께 만족시키는 out-of-fold 설계, 누수 여부 단위 테스트
- 판정 프롬프트 두 가지: 논문 부록 B의 생성형 프롬프트(제로샷), 점수 숫자 하나만 답하는 형식(LoRA)
- PEFT LoRA + TRL `SFTTrainer` 학습(정답은 평가자 개인 점수), 어댑터 병합, llm-compressor AWQ W4A16 양자화
- vLLM 서빙: 기반 모델 하나에 어댑터 15개를 올려 시드·fold별 판정, 병합 bf16과 AWQ 비교, 부하 측정
- 지표: 논문 정의의 off-by-0.5 정확도, 피어슨 r, 평가자 개인 점수와의 일치, 31개 모델 순위의 Kendall τ, 질문 수별 순위 안정성 부트스트랩, 거짓 정보 응답 감점 여부, 시드 3개 이상 구간
- 결과·비용 표 생성(`reports/RESULTS.md`), GPU 시간 추정(`reports/gpu_budget.md`)

## 아키텍처

학습과 평가는 같은 분할 위에서 돕니다.

| 구성 | 내용 |
|---|---|
| 질문 5-fold | 87개 질문을 5묶음으로 나눕니다. fold k의 test는 묶음 k 질문에 대한 모든 응답이고, 그 질문은 train에 나오지 않습니다 |
| 보류 모델 8개 | 사람 점수 순위를 8구간으로 나눠 구간마다 한 모델을 고릅니다. 이 모델들의 응답은 어느 fold의 train에도 들어가지 않습니다 |
| out-of-fold 예측 | 다섯 번 학습하면 2,506행 전체에 대해 질문을 처음 보는 상태의 예측이 생깁니다. 보류 모델의 예측은 질문과 모델을 모두 처음 보는 상태입니다 |
| 판정 형식 | 제로샷은 피드백을 쓴 뒤 `[RESULT] n`을 내고(논문 저장소와 같은 파싱), LoRA는 숫자 하나만 내며 그 토큰의 확률 분포에서 정수 점수와 기댓값을 얻습니다 |

분할 결과(fold별 행 수, 보류 모델 목록)는 [reports/dataset_profile.md](reports/dataset_profile.md)의 마지막 표에 있습니다.

판정 모델은 상업 이용이 가능한 라이선스만 썼습니다. 2026-10-03에 모델 카드 메타데이터와 LICENSE 파일로 확인했고, 커밋 해시는 [configs/models.yaml](configs/models.yaml)에 고정했습니다.

| 모델 | Hugging Face | 라이선스 | 역할 |
|---|---|---|---|
| Qwen3-4B | `Qwen/Qwen3-4B-Instruct-2507` | Apache-2.0 | 주 모델 |
| kanana-1.5-2.1b | `kakaocorp/kanana-1.5-2.1b-instruct-2505` | Apache-2.0 | 보조 모델 (시간이 남을 때) |

## 기술 스택

| 영역 | 기술 |
|---|---|
| 언어 | ![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) |
| 학습 | ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white) ![Transformers](https://img.shields.io/badge/Transformers-FFD21E?logo=huggingface&logoColor=black) ![PEFT](https://img.shields.io/badge/PEFT-LoRA-5B6770) ![TRL](https://img.shields.io/badge/TRL-SFT-5B6770) |
| 양자화·서빙 | ![llm-compressor](https://img.shields.io/badge/llm--compressor-AWQ-5B6770) ![vLLM](https://img.shields.io/badge/vLLM-5B6770) |
| 분석 | ![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white) ![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white) |
| 코드 품질 | ![pytest](https://img.shields.io/badge/pytest-0A9EDC?logo=pytest&logoColor=white) ![Ruff](https://img.shields.io/badge/Ruff-D7FF64?logo=ruff&logoColor=black) ![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?logo=githubactions&logoColor=white) |

학습·양자화 환경과 서빙 환경은 나눴습니다. vllm 0.30.0과 llmcompressor 0.14.0이 서로 다른 compressed-tensors 버전을 고정하기 때문입니다. 고정 버전은 [requirements/train.txt](requirements/train.txt), [requirements/serve.txt](requirements/serve.txt)에 있습니다.

## 실행 방법

로컬에서는 Python 3.12와 uv로 데이터 확인과 테스트까지 돌아갑니다.

```bash
make setup-dev
make data        # KUDGE 내려받기 (GitHub 커밋 09e1316, sha256 확인)
make profile     # reports/dataset_profile.{md,json}
make test        # 단위 테스트 + (torch가 있으면) 작은 랜덤 모델 학습 스모크
make lint
```

GPU 단계는 단계별 `make` 타깃으로 나눴습니다. 서버 준비, 실행 순서, 시간이 모자랄 때 줄이는 순서는 [docs/runpod_setup.md](docs/runpod_setup.md)에 정리했습니다.

| 단계 | 명령 |
|---|---|
| 제로샷 기준선 | `make zeroshot MODEL=qwen3-4b` |
| LoRA 학습 (fold 5 × 시드 3) | `make train-all MODEL=qwen3-4b` |
| LoRA 평가 | `make eval-lora MODEL=qwen3-4b` |
| 병합·양자화 | `make merge-quant MODEL=qwen3-4b` |
| 양자화 전후·부하 | `make eval-merged eval-awq MODEL=qwen3-4b GPU_NAME=…` |
| 결과·비용 표 | `make report GPU_USD_PER_HOUR=… GPU_NAME=…` |

## 평가

### 데이터 (측정 완료)

아래 수치는 모델을 쓰지 않고 데이터만으로 계산했습니다. 모두 [reports/dataset_profile.md](reports/dataset_profile.md)(`make data profile`, GitHub 커밋 `09e1316`)에서 옮겼습니다.

| 항목 | 값 |
|---|---:|
| pointwise 응답 / 질문 / 평가 대상 모델 | 2,506 / 87 / 31 |
| 모델당 응답 수 | 72~87 |
| 모든 모델이 답한 질문 | 33 |
| 거짓 정보 주입 응답 (질문 30개) | 54 |

사람 간 일치(평가자 1 대 평가자 2, 2,506행)가 이 프로젝트의 상한입니다.

| 지표 | 값 |
|---|---:|
| 완전 일치 | 47.92% |
| 1점 이내 | 83.84% |
| 피어슨 r | 0.640 |
| 2점 차이 행 | 405 |
| 평가자 1 순위 대 평가자 2 순위 (31개 모델) Kendall τ | 0.901 |

off-by-0.5 정확도의 우연 기준선은 정수 1~5 균등 추측 27.18%, 항상 1점을 주는 상수 판정 43.10%입니다. 논문 Table 3 캡션의 무작위 기준선(30.86%)과는 다르며, 차이는 [docs/kudge_verification.md](docs/kudge_verification.md)에 적었습니다.

질문 수에 따른 사람 순위 안정성(질문 복원추출 부트스트랩 1,000회):

| 질문 수 | τ(87개 전체 순위 대비) 평균 | 95% 구간 | 1위 일치 |
|---:|---:|---:|---:|
| 10 | 0.826 | 0.742~0.893 | 47.80% |
| 30 | 0.896 | 0.840~0.943 | 60.00% |
| 87 | 0.939 | 0.904~0.970 | 79.30% |

질문 20개부터 τ 2.5 백분위가 0.8을 넘고, 0.9를 넘으려면 87개 전부가 필요합니다. 같은 87개 질문에서 평가자 1 순위와 평가자 2 순위의 τ는 평균 0.867(95% 구간 0.817~0.914)입니다.

### 판정 모델 (측정 전)

| 구성 | off-by-0.5 정확도 | 피어슨 r | τ (31개 모델) | τ (보류 8개) | 거짓 정보 감지율 | 건/s | p95 | $/100만 건 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-4B 제로샷 | | | | | | | | |
| Qwen3-4B LoRA bf16 (시드 3) | | | | | | | | |
| Qwen3-4B LoRA AWQ 4bit | | | | | | | | |
| kanana-1.5-2.1b 제로샷 | | | | | | | | |
| kanana-1.5-2.1b LoRA bf16 (시드 3) | | | | | | | | |
| kanana-1.5-2.1b LoRA AWQ 4bit | | | | | | | | |

전체 표(병합 bf16 행 포함)는 `make report`가 [reports/RESULTS.md](reports/RESULTS.md)로 만듭니다. 시드 묶음 칸은 평균(최소~최대, 시드 수)으로 나오고, 시드별 t 구간은 `reports/metrics/*.json`에 남습니다.

비교용으로 논문이 발표한 수치를 인용합니다. 이 저장소에서 다시 잰 값이 아니며, 원문 표에서 옮긴 값은 [configs/paper_reference.yaml](configs/paper_reference.yaml)에 있습니다.

| 출처 (arXiv:2409.11239v2) | 판정 모델 | pointwise 정확도 | 피어슨 r |
|---|---|---:|---:|
| Table 3, 5쪽 | GPT-4o | 61.26% | 0.62 |
| Table 3, 5쪽 | Qwen2-7B-Instruct | 24.03% | 0.20 |
| Table 3, 5쪽 | Qwen2-1.5B-Instruct | 13.65% | 0.11 |
| Table 5, 7쪽 | Prometheus2-7b (영어 판정 데이터로 학습) | 46.46% | 0.43 |

## 프로젝트 구조

```text
ko-judge-lora/
├── src/ko_judge/
│   ├── data.py, profile.py        KUDGE 내려받기·확인, 데이터 프로파일
│   ├── splits.py                  질문 5-fold + 보류 모델
│   ├── prompts.py, parsing.py     판정 프롬프트, [RESULT] 파싱·점수 분포
│   ├── train_lora.py, merge.py    LoRA 학습, 어댑터 병합
│   ├── quantize.py, compat.py     AWQ W4A16, kanana 설정 검증 우회
│   ├── generation.py, predict.py  vLLM·transformers 판정
│   ├── metrics.py, ranking.py     일치도, Kendall τ, 순위 안정성 부트스트랩
│   ├── evaluate.py                out-of-fold 지표, 시드 묶음
│   └── loadtest.py, cost.py, budget.py  부하 측정, 결과·비용 표, GPU 시간 추정
├── configs/      판정 모델(라이선스·커밋 고정), LoRA 설정, 결과 표 구성, 논문 인용 수치
├── scripts/      GPU 서버 초기 설정, vLLM 실행 래퍼
├── reports/      스크립트가 만드는 결과 (데이터 프로파일·GPU 추정만 생성됨)
├── requirements/ 학습·서빙 환경 고정 버전
├── docs/         KUDGE 확인 기록, GPU 실행 가이드, 다이어그램(HTML·PNG)
└── tests/        pytest (실제 KUDGE, 합성 데이터, 가짜 판정 모델, 작은 랜덤 모델)
```

## 구현하면서 신경 쓴 점

- **질문과 모델 두 방향으로 누수를 막았습니다.** 같은 질문에 대한 다른 모델의 응답을 학습에서 보면 그 질문의 채점 감각을 외울 수 있고, 같은 모델의 다른 응답을 보면 모델 문체로 점수를 짐작할 수 있습니다. train = (그 fold 밖 질문) ∩ (보류 밖 모델)로 두 조건을 한 번의 학습에서 함께 만족시키고, 모든 fold에서 겹침이 없다는 것을 실제 데이터로 테스트합니다.
- **정답은 평균 점수가 아니라 평가자 개인 점수입니다.** 평균은 0.5 단위라 정수 판정과 맞지 않습니다. 행마다 두 평가자 중 한 명의 점수를 시드로 골라 정답으로 두고, 평가에서는 점수 토큰의 확률 기댓값을 연속 점수로 씁니다.
- **논문 지표를 그대로 쓰되 약점을 같이 적습니다.** off-by-0.5 정확도는 논문 저장소 코드와 같은 식으로 계산하고 형식 실패를 뺀 값과 포함한 값을 모두 남깁니다. 다만 항상 1점을 주는 판정도 43.10%가 나오는 지표라, 피어슨 r과 순위 τ를 같이 봅니다.
- **사람끼리의 일치를 상한으로 둡니다.** 판정과 개별 평가자의 일치를 평가자 1 대 평가자 2와 같은 척도로 재서, "사람 한 명을 다른 사람으로 바꾼 것보다 나은가"로 읽을 수 있게 했습니다.
- **순위 안정성 부트스트랩은 새로 썼습니다.** [rag-eval-harness](https://github.com/parksungyun0411/rag-eval-harness)의 순위 안정성 분석과 같은 발상(질문을 복원추출해 순위 일치를 봄)이지만, 그 구현은 모든 설정이 모든 질문에 결과를 가진 완전 행렬을 전제하고 순위 완전 일치를 기준으로 합니다. KUDGE는 모델마다 답한 질문이 다르고(공통 33개) 31개 모델의 완전 일치는 사실상 나오지 않아 Kendall τ 기준으로 다시 구현했습니다. 그 저장소가 비공개라 의존성으로 넣지 않았고, Wilson 구간 함수는 같은 식을 복제했습니다.
- **kanana 설정 문제를 우회했습니다.** transformers 5.17·5.18은 hidden 크기가 head 수로 나눠떨어지지 않는 설정을 거부해 kanana-1.5-2.1b를 불러오지 못합니다. head_dim이 명시된 경우에만 검사를 건너뛰는 패치(`compat.py`)를 학습 프로세스와 vLLM 플러그인 양쪽에 걸었고, 학습 프로세스 쪽은 테스트로 확인했습니다.

## 한계

- **모델 결과가 아직 없습니다.** GPU 단계는 실행하지 않았습니다. 로컬에서 확인한 것은 데이터 처리·분할·파싱·지표·비용 계산과, CPU에서 작은 랜덤 모델로 학습 → 병합 → 점수 분포가 2스텝 도는 것까지입니다.
- **질문이 87개뿐입니다.** fold 하나의 test는 질문 17~18개이고 보류 모델 순위 τ는 8개 모델로 계산합니다. 사람 순위조차 질문 87개 전부를 써야 τ 2.5 백분위가 0.9를 넘으므로, 판정 모델 간 τ 차이가 작으면 구분할 수 없습니다. 채점 기준(rubric)은 지식×추론 유형 조합마다 공유되므로(논문 부록 A.1) 질문을 나눠도 같은 기준문이 train과 test에 함께 나옵니다.
- **논문의 결론을 넘어서는 주장을 하지 않습니다.** 논문은 영어로 학습한 판정 모델과 보상 모델이 추가 학습 없이 한국어 평가로 옮겨 간다고 결론 내렸습니다(9절. 6.2절 Table 5에서 영어 판정 데이터로 학습한 Prometheus2-7b의 정확도는 기반 모델 Mistral-7B의 20.29%에서 46.46%로 올랐습니다). 이 프로젝트는 반대 방향, 즉 한국어 사람 점수로 같은 데이터 분포 안에서 학습한 소형 모델을 잽니다. 영어 판정 모델과 직접 비교하지 않고(논문 수치 인용만), 학습한 판정은 같은 데이터셋의 다른 질문에 대해서만 평가하므로 다른 한국어 과제로 옮겨 가는지는 알 수 없습니다. 논문 수치는 학습 없는 판정 결과라서 이 프로젝트의 LoRA 결과와 같은 조건의 비교가 아닙니다.
- **사람 점수 자체에 잡음이 있습니다.** 배포된 2,506행 중 405행은 두 평가자가 2점 다릅니다. 논문 3.2절은 이런 행을 뺐다고 적었지만 배포본에는 남아 있고, 이 프로젝트도 그대로 씁니다.
- **거짓 정보 감지 정의는 이 프로젝트의 것입니다.** 오염된 응답의 점수가 원래 응답보다 낮으면 감지로 셉니다. 논문은 피드백 글을 사람이 읽고 판단했으므로(7절) 수치를 직접 비교할 수 없고, 표본은 54건입니다.
- **vLLM 쪽 우회와 AWQ 로드는 확인하지 않았습니다.** kanana 설정 우회가 vLLM 플러그인으로 동작하는지, llm-compressor가 저장한 AWQ를 vLLM 0.30.0이 읽는지는 GPU 서버에서 처음 확인합니다. 실패하면 kanana를 빼고 Qwen3-4B 하나로 진행합니다.
- **GPU 시간은 추정입니다.** 토큰 수는 실측이지만 학습·추론 속도는 가정값이라, 첫 fold 학습 시간을 보고 다시 계산합니다.

## 라이선스

코드는 [MIT](LICENSE) 라이선스를 따릅니다. KUDGE 평가 파일은 GitHub `guijinSON/kudge`에 MIT로 배포되어 있지만, 응답 본문이 31개 모델의 출력이라 저장소에는 넣지 않고 `make data`로 고정 커밋에서 내려받습니다. HF에만 있는 원 주석 파일은 라이선스 표기가 없어 데이터 프로파일에만 씁니다. 판정 모델 가중치는 각 모델의 라이선스(Apache-2.0)를 따릅니다. 근거는 [docs/kudge_verification.md](docs/kudge_verification.md)에 있습니다.
