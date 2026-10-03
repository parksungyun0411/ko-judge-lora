# GPU 임대 서버 실행 가이드

GPU 단계는 아직 한 번도 실행하지 않았습니다. 아래 순서와 시간은 코드와 토큰 수를 기준으로 세운 계획이며, 첫 실행에서 막히는 곳은 이 문서에 고쳐 적습니다.

## 서버 조건

| 항목 | 값 |
|---|---|
| GPU | VRAM 24GB 1장 (RTX 4090 기준으로 계획) |
| 이미지 | CUDA 12.x 드라이버가 있는 Ubuntu (RunPod PyTorch 템플릿 등) |
| 디스크 | 영구 볼륨 150GB 이상 — 기반 모델 2개, 병합 bf16 10개, AWQ 10개 |
| 네트워크 | Hugging Face, GitHub 접근 (모델·데이터 모두 공개, 토큰 불필요) |

## 예상 GPU 시간 (추정)

[reports/gpu_budget.md](../reports/gpu_budget.md)가 `make budget`으로 만든 추정입니다. 프롬프트 토큰 수는 실제 데이터와 실제 토크나이저로 셌고(Qwen3-4B 기준 2,506행 합계 5,165,208토큰, 평균 2,061), 학습·추론 속도는 가정값입니다.

| 범위 | 추정 시간 |
|---|---:|
| 두 모델 전체 (fold 5 × 시드 3, 여유 20% 포함) | 약 26.3h |
| Qwen3-4B 하나만 (시간이 모자랄 때) | 약 16.9h |

비용은 `시간 × 시간당 요금`입니다. 대여 직전에 요금표를 확인하고 승인받은 뒤 시작합니다. 실제 요금은 `make report GPU_USD_PER_HOUR=…`에 넣습니다.

## 실행 순서

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

1. kanana-1.5-2.1b를 뺍니다. Qwen3-4B 하나로 모든 비교(제로샷 대 LoRA, 양자화 전후, 순위 재현)가 성립합니다.
2. epoch를 2에서 1로 줄입니다(`configs/lora_qwen3-4b.yaml`). README에 바꾼 사실을 적습니다.
3. 시드를 3에서 2로 줄이지는 않습니다. 시드 간 변동이 이 프로젝트의 주요 수치라서, 그 전에 fold 수를 바꾸는 쪽도 하지 않습니다(분할이 달라지면 데이터 프로파일과 맞지 않습니다).

## 먼저 확인할 것

1. **kanana 설정 로드.** transformers 5.17·5.18은 kanana-1.5-2.1b 설정(hidden 1792, head 24, head_dim 128)을 거부합니다. 학습·병합·양자화는 `ko_judge.compat`로 우회하고, vLLM은 `vllm.general_plugins` 진입점으로 같은 우회를 겁니다. 서버에서 `make zeroshot MODEL=kanana-1.5-2.1b`가 설정 오류로 멈추면 플러그인이 불리지 않은 것이니 kanana를 빼고 진행합니다.
2. **AWQ 체크포인트 로드.** 학습 환경의 llm-compressor 0.14.0은 compressed-tensors 0.19.0으로 저장하고 vLLM 0.30.0은 0.17.0을 씁니다. `make eval-awq`의 첫 fold에서 vLLM이 읽지 못하면 `.venv-train`에 `llmcompressor==0.12.0`(compressed-tensors 0.17.1, transformers ≤5.10.1)을 설치해 양자화만 다시 합니다.
3. **logprob 토큰.** 점수만 모드는 첫 생성 토큰의 상위 20개 logprob에서 '1'~'5'를 찾습니다. 첫 fold 예측 파일에서 `dist`가 None인 행이 있으면 채팅 템플릿이 숫자 앞에 다른 토큰을 넣는 것이니 원인을 보고 고칩니다.
