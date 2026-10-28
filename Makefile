# 단계별 실행. GPU 단계는 scripts/runpod_bootstrap.sh로 .venv-train·.venv-serve를 만든 뒤 쓴다.
# 예: make train-all MODEL=qwen3-4b
SHELL := /bin/bash
.DEFAULT_GOAL := help

PY       ?= $(firstword $(wildcard .venv/bin/python .venv-serve/bin/python) python3)
TRAIN_PY ?= .venv-train/bin/python
SERVE_PY ?= .venv-serve/bin/python
MODEL    ?= qwen3-4b
FOLDS    ?= 0 1 2 3 4
SEEDS    ?= 0 1 2
FOLD     ?= 0
SEED     ?= 0
GPU_NAME ?=
GPU_USD_PER_HOUR ?=
PRICE_NOTE ?=
CONCURRENCY ?= 32
# bf16이 없는 GPU(T4)용: CONFIG=configs/t4/lora_$(MODEL).yaml VLLM_ARGS="--dtype float16" MERGE_DTYPE=float16 PREC=fp16
#   QUANT_ARGS="--num-samples 32" (보정 샘플 256개·64개는 T4 15GB를 넘었다, docs/kaggle_t4.md)
CONFIG   ?= configs/lora_$(MODEL).yaml
VLLM_ARGS ?=
MERGE_DTYPE ?= bfloat16
QUANT_ARGS ?=
PREC     ?= bf16

HF_ID    = $(shell $(PY) -m ko_judge.config $(MODEL) hf_id)
REVISION = $(shell $(PY) -m ko_judge.config $(MODEL) revision)
LORA_DIR = outputs/lora/$(MODEL)
LORA_MODULES = $(foreach s,$(SEEDS),$(foreach f,$(FOLDS),$(MODEL)-f$(f)-s$(s)=$(LORA_DIR)/fold$(f)_seed$(s)))

.PHONY: help setup-dev test lint smoke data profile budget setup-gpu zeroshot train train-all \
        eval-lora eval-seeds-fold0 merge-quant eval-merged eval-awq report checks

help:
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

# ── 로컬 ────────────────────────────────────────────────
setup-dev: ## 로컬 개발 환경 (.venv, Python 3.12)
	uv venv --python 3.12 .venv
	uv pip install --python .venv -e ".[dev]"

test: ## pytest (스모크는 torch가 있을 때, 데이터 테스트는 data/kudge가 있을 때만)
	$(PY) -m pytest -q

lint: ## ruff
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

smoke: ## 작은 랜덤 모델로 학습→병합→점수 경로 확인 (CPU)
	$(PY) -m pytest -q -m smoke

data: ## KUDGE 내려받기 (고정 커밋, sha256 확인) → data/kudge/
	$(PY) -m ko_judge.data

profile: ## 데이터 프로파일 → reports/dataset_profile.{md,json}
	$(PY) -m ko_judge.profile

budget: ## GPU 시간 추정 (토큰 수는 실측, 속도는 가정) → reports/gpu_budget.{md,json}
	$(PY) -m ko_judge.budget

# ── GPU ────────────────────────────────────────────────
setup-gpu: ## GPU 서버 환경 (.venv-train, .venv-serve) + 데이터
	bash scripts/runpod_bootstrap.sh

zeroshot: ## 기반 모델 생성형 제로샷, 2,506행 전체 (MODEL=키)
	scripts/with_vllm.sh $(HF_ID) $(MODEL) --revision $(REVISION) $(VLLM_ARGS) -- bash -c '\
	  $(SERVE_PY) -m ko_judge.predict --mode judge --served $(MODEL) --fold all --tag zeroshot_$(MODEL) && \
	  $(SERVE_PY) -m ko_judge.evaluate run --tag zeroshot_$(MODEL)'

train: ## LoRA fold 하나 (MODEL FOLD SEED)
	$(TRAIN_PY) -m ko_judge.train_lora --config $(CONFIG) --fold $(FOLD) --seed $(SEED)

train-all: ## LoRA 전체: FOLDS × SEEDS
	for s in $(SEEDS); do for f in $(FOLDS); do \
	  $(TRAIN_PY) -m ko_judge.train_lora --config $(CONFIG) --fold $$f --seed $$s || exit 1; \
	done; done

eval-lora: ## 기반 모델 + 어댑터 전부를 한 서버로 → 시드별 지표 + 시드 묶음
	scripts/with_vllm.sh $(HF_ID) $(MODEL) --revision $(REVISION) $(VLLM_ARGS) --enable-lora --max-lora-rank 16 \
	  --max-loras 4 --lora-modules $(LORA_MODULES) -- bash -c '\
	  for s in $(SEEDS); do for f in $(FOLDS); do \
	    $(SERVE_PY) -m ko_judge.predict --mode score --served $(MODEL)-f$$f-s$$s --fold $$f \
	      --tag lora_$(MODEL)_$(PREC)_s$$s || exit 1; \
	  done; $(SERVE_PY) -m ko_judge.evaluate run --tag lora_$(MODEL)_$(PREC)_s$$s || exit 1; done'
	$(PY) -m ko_judge.evaluate seeds --group lora_$(MODEL)_$(PREC) \
	  --tags $(foreach s,$(SEEDS),lora_$(MODEL)_$(PREC)_s$(s))

eval-seeds-fold0: ## 축소안: fold 0 시드별 어댑터로 학습 시드 변동 (SEEDS=0 1 2)
	scripts/with_vllm.sh $(HF_ID) $(MODEL) --revision $(REVISION) $(VLLM_ARGS) --enable-lora --max-lora-rank 16 \
	  --max-loras 4 --lora-modules $(foreach s,$(SEEDS),$(MODEL)-f0-s$(s)=$(LORA_DIR)/fold0_seed$(s)) \
	  -- bash -c 'for s in $(SEEDS); do \
	    $(SERVE_PY) -m ko_judge.predict --mode score --served $(MODEL)-f0-s$$s --fold 0 \
	      --tag lora_$(MODEL)_f0only_s$$s || exit 1; \
	    $(SERVE_PY) -m ko_judge.evaluate run --tag lora_$(MODEL)_f0only_s$$s --only-fold 0 || exit 1; done'
	$(PY) -m ko_judge.evaluate seeds --group lora_$(MODEL)_fold0_seeds \
	  --tags $(foreach s,$(SEEDS),lora_$(MODEL)_f0only_s$(s)_fold0)

merge-quant: ## 시드 0의 fold별 어댑터 병합 → AWQ 4bit
	for f in $(FOLDS); do \
	  $(TRAIN_PY) -m ko_judge.merge --model $(MODEL) --adapter $(LORA_DIR)/fold$${f}_seed0 \
	    --out outputs/merged/$(MODEL)/fold$${f}_seed0 --dtype $(MERGE_DTYPE) || exit 1; \
	  $(TRAIN_PY) -m ko_judge.quantize --model-dir outputs/merged/$(MODEL)/fold$${f}_seed0 \
	    --out outputs/awq/$(MODEL)/fold$${f}_seed0 --fold $$f $(QUANT_ARGS) || exit 1; \
	done

eval-merged: ## 병합 모델(MERGE_DTYPE)을 fold별로 서빙해 판정, fold 0에서 부하 측정
	for f in $(FOLDS); do \
	  scripts/with_vllm.sh outputs/merged/$(MODEL)/fold$${f}_seed0 $(MODEL)-merged-f$$f $(VLLM_ARGS) -- bash -c "\
	    $(SERVE_PY) -m ko_judge.predict --mode score --served $(MODEL)-merged-f$$f --fold $$f \
	      --tag lora_$(MODEL)_merged_s0 && \
	    if [ $$f = 0 ]; then $(SERVE_PY) -m ko_judge.loadtest --served $(MODEL)-merged-f0 \
	      --tag lora_$(MODEL)_merged --concurrency $(CONCURRENCY) --gpu '$(GPU_NAME)'; fi" || exit 1; \
	done
	$(PY) -m ko_judge.evaluate run --tag lora_$(MODEL)_merged_s0

eval-awq: ## AWQ 4bit를 fold별로 서빙해 판정, fold 0에서 부하 측정
	for f in $(FOLDS); do \
	  scripts/with_vllm.sh outputs/awq/$(MODEL)/fold$${f}_seed0 $(MODEL)-awq-f$$f $(VLLM_ARGS) -- bash -c "\
	    $(SERVE_PY) -m ko_judge.predict --mode score --served $(MODEL)-awq-f$$f --fold $$f \
	      --tag lora_$(MODEL)_awq_s0 && \
	    if [ $$f = 0 ]; then $(SERVE_PY) -m ko_judge.loadtest --served $(MODEL)-awq-f0 \
	      --tag lora_$(MODEL)_awq --concurrency $(CONCURRENCY) --gpu '$(GPU_NAME)'; fi" || exit 1; \
	done
	$(PY) -m ko_judge.evaluate run --tag lora_$(MODEL)_awq_s0

report: ## reports/RESULTS.md 생성 (GPU_USD_PER_HOUR=값 GPU_NAME=이름 PRICE_NOTE=요금 근거)
	$(PY) -m ko_judge.cost --gpu-name "$(GPU_NAME)" --price-note "$(PRICE_NOTE)" \
	  $(if $(GPU_USD_PER_HOUR),--gpu-usd-per-hour $(GPU_USD_PER_HOUR),)

checks: ## 저장소의 예측·실행 기록에서 비교 수치 → reports/run_checks.json
	$(PY) -m ko_judge.run_checks
