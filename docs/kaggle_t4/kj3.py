# ko-judge-lora Kaggle T4 x2 (kj3, 2026-10-23). kanana-1.5-2.1b fp16, 시드 2 5 fold.
#   GPU0: 시드 2 fold 0 → 2 → 4
#   GPU1: 시드 0 어댑터 병합·AWQ 채점(fold 2 3 4) → 시드 2 fold 1 → 3
import subprocess

SH = r'''
set -u
START=$(date +%s); LIMIT=$((START + 11*3600 + 1200))
W=/kaggle/working; mkdir -p $W/logs $W/out
export HF_HOME=/tmp/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $W/logs/steps.log; }
git clone -q https://github.com/parksungyun0411/ko-judge-lora /tmp/kj && cd /tmp/kj && git checkout -q a2b5d9004141e72c0f22b36671254697ce474551 || { log "CLONE_FAILED"; exit 1; }
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv | tee $W/logs/gpu.txt
M=kanana-1.5-2.1b; CFG=configs/t4/lora_$M.yaml; HF=kakaocorp/kanana-1.5-2.1b-instruct-2505
SEED=2; TAG=lora_${M}_fp16_s$SEED
uv venv -q --python 3.12 .venv-train && uv pip install -q --python .venv-train -r requirements/train.txt && uv pip install -q --python .venv-train --no-deps -e . || { log "SETUP_TRAIN_FAILED"; exit 1; }
uv venv -q --python 3.12 .venv-serve && uv pip install -q --python .venv-serve -r requirements/serve.txt && uv pip install -q --python .venv-serve --no-deps -e . || { log "SETUP_SERVE_FAILED"; exit 1; }
.venv-train/bin/python -m ko_judge.data > $W/logs/data.log 2>&1 || { log "DATA_FAILED"; exit 1; }
REV=$(.venv-train/bin/python -m ko_judge.config $M revision)
.venv-train/bin/python -c "from huggingface_hub import snapshot_download as s; print(s('$HF', revision='$REV'))" > $W/logs/download.log 2>&1
# 시드 0 어댑터 (kj-k1 출력)
for k in 2 3 4; do
  t=$(find /kaggle/input -name "adapter_${M}_fold${k}_seed0.tgz" | head -1)
  [ -n "$t" ] && tar xzf "$t" -C /tmp/kj || log "NO_ADAPTER fold$k"
done
log "SETUP_DONE rev=$REV sha=$(git rev-parse --short HEAD) seed=$SEED"

pack() {
  ( cd /tmp/kj
    for p in outputs/predictions reports/metrics reports/throughput; do [ -d "$p" ] && mkdir -p "$W/out/$p" && cp -r "$p/." "$W/out/$p/"; done
    for f in outputs/lora/$M/fold*_seed$SEED/train_summary.json; do [ -f "$f" ] && mkdir -p "$W/out/$(dirname "$f")" && cp "$f" "$W/out/$f"; done
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
serve() {  # serve <port> <logdir> <model> <name> -- <cmd...>
  local port=$1 ld=$2; shift 2
  env PORT=$port LOG_DIR=$ld VLLM=.venv-serve/bin/vllm scripts/with_vllm.sh "$@"
}
fold() {  # fold <gpu> <k> <port>: 시드 SEED 학습 → 어댑터 서빙 채점
  local g=$1 k=$2 port=$3
  step $g train_f$k 11500 .venv-train/bin/python -m ko_judge.train_lora --config $CFG --fold $k --seed $SEED || return 1
  tar czf $W/adapter_${M}_fold${k}_seed$SEED.tgz --exclude='checkpoint-*' -C /tmp/kj outputs/lora/$M/fold${k}_seed$SEED
  step $g score_f$k 1500 serve $port outputs/logs/score_f$k $HF $M --revision $REV --dtype float16 --enable-lora \
    --max-lora-rank 16 --max-loras 1 --lora-modules $M-f$k-s$SEED=outputs/lora/$M/fold${k}_seed$SEED -- \
    .venv-serve/bin/python -m ko_judge.predict --mode score --served $M-f$k-s$SEED --fold $k --tag $TAG \
    --base-url http://localhost:$port/v1
  step $g eval_f$k 300 .venv-train/bin/python -m ko_judge.evaluate run --tag $TAG --only-fold $k
  return 0
}
mq() {  # mq <gpu> <k> <port> <부하측정 0|1>: 시드 0 어댑터 병합(fp16) → 채점 → AWQ → 채점
  local g=$1 k=$2 port=$3 lt=$4 d=outputs/merged/$M/fold${k}_seed0 q=outputs/awq/$M/fold${k}_seed0
  step $g merge_f$k 600 .venv-train/bin/python -m ko_judge.merge --model $M --adapter outputs/lora/$M/fold${k}_seed0 \
    --out $d --dtype float16 || return 1
  local extra="true"
  [ "$lt" = 1 ] && extra=".venv-serve/bin/python -m ko_judge.loadtest --served $M-merged-f$k --tag lora_${M}_merged --concurrency 32 --gpu 'Tesla T4' --base-url http://localhost:$port/v1"
  step $g score_merged_f$k 4000 serve $port outputs/logs/merged_f$k $d $M-merged-f$k --dtype float16 -- bash -c "\
    .venv-serve/bin/python -m ko_judge.predict --mode score --served $M-merged-f$k --fold $k --tag lora_${M}_merged_s0 \
      --base-url http://localhost:$port/v1 && $extra"
  step $g quantize_f$k 2400 .venv-train/bin/python -m ko_judge.quantize --model-dir $d --out $q --fold $k || return 1
  extra="true"
  [ "$lt" = 1 ] && extra=".venv-serve/bin/python -m ko_judge.loadtest --served $M-awq-f$k --tag lora_${M}_awq --concurrency 32 --gpu 'Tesla T4' --base-url http://localhost:$port/v1"
  step $g score_awq_f$k 4000 serve $port outputs/logs/awq_f$k $q $M-awq-f$k --dtype float16 -- bash -c "\
    .venv-serve/bin/python -m ko_judge.predict --mode score --served $M-awq-f$k --fold $k --tag lora_${M}_awq_s0 \
      --base-url http://localhost:$port/v1 && $extra"
  rm -rf $d $q
  return 0
}
gpu0() { export CUDA_VISIBLE_DEVICES=0; for k in 0 2 4; do fold 0 $k 8000; done; }
gpu1() { export CUDA_VISIBLE_DEVICES=1; mq 1 2 8001 0; mq 1 3 8001 0; mq 1 4 8001 0; for k in 1 3; do fold 1 $k 8001; done; }
( sleep $((LIMIT - $(date +%s))); log "HARD_STOP"; pkill -f ko_judge; pkill -f "vllm serve"; sleep 20; pack ) &
WD=$!
gpu0 & P0=$!
gpu1 & P1=$!
wait $P0; wait $P1
kill $WD 2>/dev/null
step c eval_seed_all 600 .venv-train/bin/python -m ko_judge.evaluate run --tag $TAG
pack
du -sh $W/out $W/*.tgz 2>/dev/null | tee -a $W/logs/steps.log
log "ALL_DONE"
'''
subprocess.run(["bash", "-c", SH])
print(open("/kaggle/working/logs/steps.log").read())
