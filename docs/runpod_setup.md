# GPU 임대 서버 실행 가이드

이 문서는 처음 계획(RTX 4090 임대, Qwen3-4B 주 모델)의 절차이고, 이 경로로는 실행하지 않았습니다. 실제 GPU 단계는 Kaggle 무료 T4 두 장에서 kanana-1.5-2.1b만 fp16으로 돌렸고, 절차·결과·실행 중 생긴 문제는 [kaggle_t4.md](kaggle_t4.md)에 있습니다. 아래 순서와 시간은 코드와 토큰 수를 기준으로 세운 계획 그대로 두었습니다.

## 서버 조건

| 항목 | 값 |
|---|---|
| GPU | VRAM 24GB 1장 (RTX 4090 기준으로 계획) |
| 이미지 | CUDA 12.x 드라이버가 있는 Ubuntu (RunPod PyTorch 템플릿 등) |
| 디스크 | 영구 볼륨 150GB 이상 (기반 모델 2개, 병합 bf16 10개, AWQ 10개) |
| 네트워크 | Hugging Face, GitHub 접근 (모델·데이터 모두 공개, 토큰 불필요) |

## 예상 GPU 시간 (추정)

[reports/gpu_budget.md](../reports/gpu_budget.md)가 `make budget`으로 만든 추정입니다. 프롬프트 토큰 수는 실제 데이터와 실제 토크나이저로 셌고(Qwen3-4B 기준 2,506행 합계 5,165,208토큰, 평균 2,061), 학습·추론 속도는 가정값입니다.

| 범위 | 추정 시간 |
|---|---:|
| 두 모델 전체 (fold 5 × 시드 3, 여유 20% 포함) | 약 26.0h |
| Qwen3-4B 하나만, fold 5 × 시드 3 | 약 16.8h |
| 축소안: Qwen3-4B 하나, 시드 0 × fold 5 + fold 0에서 시드 1·2 (학습 7회) | 약 10.1h |

비용은 `시간 × 시간당 요금`입니다. 대여 직전에 요금표를 확인하고 승인받은 뒤 시작합니다. 실제 요금은 `make report GPU_USD_PER_HOUR=…`에 넣습니다.

## 축소안 (약 10h, 추정)

기본값(fold 5 × 시드 3, 두 모델)은 바꾸지 않았습니다. 시간을 10시간 안팎으로 맞추려면 아래처럼 돌립니다.

```bash
make zeroshot MODEL=qwen3-4b
make train-all MODEL=qwen3-4b SEEDS=0               # fold 5 × 시드 0
make train MODEL=qwen3-4b FOLD=0 SEED=1
make train MODEL=qwen3-4b FOLD=0 SEED=2
make eval-lora MODEL=qwen3-4b SEEDS=0               # 2,506행 out-of-fold, 시드 0
make eval-seeds-fold0 MODEL=qwen3-4b SEEDS="0 1 2"  # fold 0에서만 시드 간 변동
make merge-quant eval-merged eval-awq MODEL=qwen3-4b GPU_NAME="RTX 4090"
make report GPU_USD_PER_HOUR=<실제 요금> GPU_NAME="RTX 4090"
```

축소안으로 말할 수 있는 것:

- 시드 0 한 번의 out-of-fold 예측으로 2,506행 전체의 사람 점수 일치, 31개·보류 8개 모델 순위 τ, 질문 수별 부트스트랩 구간(질문 표본에 따른 변동)
- 제로샷 대 LoRA, 병합 bf16 대 AWQ 4bit의 차이와 건당 비용
- 학습 시드에 따른 변동. 단, fold 0(test 질문 18개, 499행)에서 시드 3개로만

말할 수 없는 것:

- 전체 데이터 지표의 시드 간 구간. 그래서 양자화 전후 차이나 제로샷 대비 개선이 학습 시드에 따른 흔들림보다 큰지를 전체 데이터 기준으로 판단할 수 없습니다. fold 0의 시드 변동으로 짐작만 할 수 있습니다.
- 판정 모델 간 비교(kanana-1.5-2.1b 제외)

## 실행 순서 (기본안)

```bash
# ── 서버 준비 ──
git clone <이 저장소> && cd ko-judge-lora
bash scripts/runpod_bootstrap.sh       # .venv-train, .venv-serve, 데이터 내려받기(sha256 확인)

# ── 1. 첫 fold로 속도 확인 (중요) ──
make train MODEL=qwen3-4b FOLD=0 SEED=0
# 로그의 스텝 시간으로 학습 1회 시간을 재고, src/ko_judge/budget.py의 가정값을 고쳐 `make budget`을 다시 낸다.
# 1회가 추정의 1.5배를 넘으면 아래 '줄이는 순서'를 따른다.

# ── 2. 주 모델 Qwen3-4B ──
make zeroshot MODEL=qwen3-4b
make train-all MODEL=qwen3-4b           # fold 5 × 시드 3 (fold0_seed0은 이미 있으면 다시 학습한다)
make eval-lora MODEL=qwen3-4b           # 어댑터 15개를 한 서버로, 시드별 지표 + 시드 묶음
make merge-quant MODEL=qwen3-4b         # 시드 0 fold별 병합 → AWQ
make eval-merged MODEL=qwen3-4b GPU_NAME="RTX 4090"
make eval-awq MODEL=qwen3-4b GPU_NAME="RTX 4090"

# ── 3. 보조 모델 kanana-1.5-2.1b (시간이 남을 때) ──
make zeroshot train-all eval-lora merge-quant MODEL=kanana-1.5-2.1b
make eval-merged eval-awq MODEL=kanana-1.5-2.1b GPU_NAME="RTX 4090"

# ── 4. 결과 표 ──
make report GPU_USD_PER_HOUR=<실제 요금> GPU_NAME="RTX 4090"
# reports/ 와 outputs/lora/*/*/train_summary.json 을 내려받고 서버를 끈다
```

## 줄이는 순서 (시간이 모자랄 때)

1. kanana-1.5-2.1b를 뺍니다(약 16.8h). Qwen3-4B 하나로 제로샷 대 LoRA, 양자화 전후, 순위 재현 비교가 모두 성립하고, 시드 3개 구간도 그대로 남습니다.
2. 그래도 모자라면 위 축소안(약 10.1h)으로 갑니다. 전체 데이터 지표는 시드 하나가 되고, 시드 변동은 fold 0에서만 봅니다.
3. epoch를 2에서 1로 줄이는 것은 마지막 수단입니다(`configs/lora_qwen3-4b.yaml`). 바꾸면 README에 적습니다.
4. fold 수는 바꾸지 않습니다. 분할이 달라지면 데이터 프로파일·보류 모델과 맞지 않습니다.

## 먼저 확인할 것

1. **kanana 설정 로드.** transformers 5.17·5.18은 kanana-1.5-2.1b 설정(hidden 1792, head 24, head_dim 128)을 거부합니다. 학습·병합·양자화는 `ko_judge.compat`로 우회하고, vLLM은 `vllm.general_plugins` 진입점으로 같은 우회를 겁니다. 서버에서 `make zeroshot MODEL=kanana-1.5-2.1b`가 설정 오류로 멈추면 플러그인이 불리지 않은 것이니 kanana를 빼고 진행합니다. T4 실행에서는 vLLM이 플러그인으로 kanana 기반 모델·어댑터·병합 모델을 모두 읽었습니다.
2. **AWQ 체크포인트 로드.** 학습 환경의 llm-compressor 0.14.0은 compressed-tensors 0.19.0으로 저장하고 vLLM 0.30.0은 0.17.0을 씁니다. `make eval-awq`의 첫 fold에서 vLLM이 읽지 못하면 `.venv-train`에 `llmcompressor==0.12.0`(compressed-tensors 0.17.1, transformers ≤5.10.1)을 설치해 양자화만 다시 합니다. T4 실행에서는 llm-compressor 0.14.0으로 저장한 체크포인트를 vLLM 0.30.0이 `MarlinLinearKernel`로 읽어 다섯 fold 모두 판정했습니다([kaggle_t4/vllm_awq_f0.log](kaggle_t4/vllm_awq_f0.log)).
3. **logprob 토큰.** 점수만 모드는 첫 생성 토큰의 상위 20개 logprob에서 '1'~'5'를 찾습니다. 첫 fold 예측 파일에서 `dist`가 None인 행이 있으면 채팅 템플릿이 숫자 앞에 다른 토큰을 넣는 것이니 원인을 보고 고칩니다. T4 점검에서는 상위 토큰에 섞인 '①'을 숫자로 읽다 오류가 나서 ASCII 숫자만 받도록 고쳤고(`a2b5d90`), 그 뒤 LoRA 판정의 형식 실패는 0행이었습니다.
