# ko-judge-lora Kaggle T4 x2 본 실행 (2026-10-16). kanana-1.5-2.1b, fp16, 시드 0.
#   GPU0: fold 0 → 2 → 4 (LoRA 학습 직후 그 fold의 test를 점수 모드로 채점)
#   GPU1: 제로샷 판정(2,506행 + 허위정보 항목, 5 fold 전부) → fold 1 → 3
# 단계마다 남은 시간이 예상 시간보다 짧으면 시작하지 않는다. 11시간 20분에 강제 종료하고 결과를 모은다.
# 점검(kj-smoke): 최장 프롬프트 배치 2 최대 8.3GiB, 학습 55 s/스텝(fold 약 2.9h), 제로샷 fold 0 약 42분.
import subprocess

SH = r'''
set -u
START=$(date +%s); LIMIT=$((START + 11*3600 + 1200))
W=/kaggle/working; mkdir -p $W/logs $W/out
export HF_HOME=/tmp/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $W/logs/steps.log; }
git clone -q https://github.com/parksungyun0411/ko-judge-lora /tmp/kj && cd /tmp/kj && git checkout -q __SHA__ || { log "CLONE_FAILED"; exit 1; }
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv | tee $W/logs/gpu.txt
M=kanana-1.5-2.1b; CFG=configs/t4/lora_$M.yaml; TAG=lora_${M}_fp16_s0; HF=kakaocorp/kanana-1.5-2.1b-instruct-2505

uv venv -q --python 3.12 .venv-train && uv pip install -q --python .venv-train -r requirements/train.txt && uv pip install -q --python .venv-train --no-deps -e . || { log "SETUP_TRAIN_FAILED"; exit 1; }
uv venv -q --python 3.12 .venv-serve && uv pip install -q --python .venv-serve -r requirements/serve.txt && uv pip install -q --python .venv-serve --no-deps -e . || { log "SETUP_SERVE_FAILED"; exit 1; }
uv pip freeze --python .venv-train > $W/logs/freeze_train.txt; uv pip freeze --python .venv-serve > $W/logs/freeze_serve.txt
.venv-train/bin/python -m ko_judge.data > $W/logs/data.log 2>&1 || { log "DATA_FAILED"; exit 1; }
REV=$(.venv-train/bin/python -m ko_judge.config $M revision)
.venv-train/bin/python -c "from huggingface_hub import snapshot_download as s; print(s('$HF', revision='$REV'))" > $W/logs/download.log 2>&1
log "SETUP_DONE rev=$REV sha=$(git rev-parse --short HEAD)"

pack() {
  ( cd /tmp/kj
    for p in outputs/predictions reports/metrics; do [ -d "$p" ] && mkdir -p "$W/out/$p" && cp -r "$p/." "$W/out/$p/"; done
    for f in outputs/lora/$M/fold*_seed0/train_summary.json; do [ -f "$f" ] && mkdir -p "$W/out/$(dirname "$f")" && cp "$f" "$W/out/$f"; done
    [ -d outputs/logs ] && mkdir -p $W/out/vllm_logs && cp -r outputs/logs/. $W/out/vllm_logs/ )
  return 0
}
step() {  # step <gpu> <이름> <필요 초> <명령...>
  local g=$1 name=$2 need=$3; shift 3
  local left=$(( LIMIT - $(date +%s) ))
  if [ "$left" -lt "$need" ]; then log "[gpu$g] SKIP $name (남은 ${left}s < 필요 ${need}s)"; return 1; fi
  log "[gpu$g] START $name"
  "$@" > "$W/logs/$name.log" 2>&1; local rc=$?
  if [ $rc -eq 0 ]; then log "[gpu$g] OK $name"; else log "[gpu$g] FAILED $name rc=$rc"; fi
  pack; return $rc
}
fold() {  # fold <gpu> <k> <port>
  local g=$1 k=$2 port=$3
  step $g train_f$k 11500 .venv-train/bin/python -m ko_judge.train_lora --config $CFG --fold $k --seed 0 || return 1
  tar czf $W/adapter_${M}_fold${k}_seed0.tgz --exclude='checkpoint-*' -C /tmp/kj outputs/lora/$M/fold${k}_seed0
  step $g score_f$k 1500 env PORT=$port LOG_DIR=outputs/logs/score_f$k VLLM=.venv-serve/bin/vllm \
    scripts/with_vllm.sh $HF $M --revision $REV --dtype float16 --enable-lora --max-lora-rank 16 --max-loras 1 \
    --lora-modules $M-f$k-s0=outputs/lora/$M/fold${k}_seed0 -- \
    .venv-serve/bin/python -m ko_judge.predict --mode score --served $M-f$k-s0 --fold $k --tag $TAG \
    --base-url http://localhost:$port/v1
  step $g eval_f$k 300 .venv-train/bin/python -m ko_judge.evaluate run --tag $TAG --only-fold $k
  return 0
}
gpu0() {
  export CUDA_VISIBLE_DEVICES=0
  fold 0 0 8000; fold 0 2 8000; fold 0 4 8000
}
gpu1() {
  export CUDA_VISIBLE_DEVICES=1
  step 1 zeroshot 14400 env PORT=8001 LOG_DIR=outputs/logs/zeroshot VLLM=.venv-serve/bin/vllm \
    scripts/with_vllm.sh $HF $M --revision $REV --dtype float16 -- \
    .venv-serve/bin/python -m ko_judge.predict --mode judge --served $M --fold all --tag zeroshot_$M \
    --base-url http://localhost:8001/v1
  step 1 eval_zeroshot 600 .venv-train/bin/python -m ko_judge.evaluate run --tag zeroshot_$M
  fold 1 1 8001; fold 1 3 8001
}
( sleep $((LIMIT - $(date +%s))); log "HARD_STOP"; pkill -f ko_judge; pkill -f "vllm serve"; sleep 20; pack ) &
WD=$!
gpu0 & P0=$!
gpu1 & P1=$!
wait $P0; wait $P1
kill $WD 2>/dev/null
step c eval_lora_all 600 .venv-train/bin/python -m ko_judge.evaluate run --tag $TAG
pack
du -sh $W/out $W/*.tgz 2>/dev/null | tee -a $W/logs/steps.log
log "ALL_DONE"
'''
SH = SH.replace("__SHA__", "a2b5d9004141e72c0f22b36671254697ce474551")
subprocess.run(["bash", "-c", SH])
print(open("/kaggle/working/logs/steps.log").read())
