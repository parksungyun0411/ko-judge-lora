# ko-judge-lora Kaggle T4 x2 (kj4, 2026-10-26). kanana-1.5-2.1b 시드 0 어댑터 병합(fp16) → AWQ → 채점.
#   kj2·kj3의 AWQ 보정이 T4 메모리를 넘어(샘플 256개, 길이 중앙값 1,901토큰) 다섯 fold 모두 실패했다.
#   보정 샘플을 64개로 줄이고(실패하면 32개), 병합 모델 채점은 kj2·kj3에서 끝났으므로 하지 않는다.
#   GPU0: fold 0(부하 측정 포함) → 2 → 4, GPU1: fold 1 → 3
import subprocess

SH = r'''
set -u
START=$(date +%s); LIMIT=$((START + 11*3600 + 1200))
W=/kaggle/working; mkdir -p $W/logs $W/out
export HF_HOME=/tmp/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $W/logs/steps.log; }
git clone -q https://github.com/parksungyun0411/ko-judge-lora /tmp/kj && cd /tmp/kj && git checkout -q a2b5d9004141e72c0f22b36671254697ce474551 || { log "CLONE_FAILED"; exit 1; }
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv | tee $W/logs/gpu.txt
M=kanana-1.5-2.1b; CFG=configs/t4/lora_$M.yaml; HF=kakaocorp/kanana-1.5-2.1b-instruct-2505
SEED=0
uv venv -q --python 3.12 .venv-train && uv pip install -q --python .venv-train -r requirements/train.txt && uv pip install -q --python .venv-train --no-deps -e . || { log "SETUP_TRAIN_FAILED"; exit 1; }
uv venv -q --python 3.12 .venv-serve && uv pip install -q --python .venv-serve -r requirements/serve.txt && uv pip install -q --python .venv-serve --no-deps -e . || { log "SETUP_SERVE_FAILED"; exit 1; }
.venv-train/bin/python -m ko_judge.data > $W/logs/data.log 2>&1 || { log "DATA_FAILED"; exit 1; }
REV=$(.venv-train/bin/python -m ko_judge.config $M revision)
.venv-train/bin/python -c "from huggingface_hub import snapshot_download as s; print(s('$HF', revision='$REV'))" > $W/logs/download.log 2>&1
# 시드 0 어댑터 (kj-k1 출력)
for k in 0 1 2 3 4; do
  t=$(find /kaggle/input -name "adapter_${M}_fold${k}_seed0.tgz" | head -1)
  [ -n "$t" ] && tar xzf "$t" -C /tmp/kj || log "NO_ADAPTER fold$k"
done
log "SETUP_DONE rev=$REV sha=$(git rev-parse --short HEAD) seed=$SEED"

pack() {
  ( cd /tmp/kj
    for p in outputs/predictions reports/metrics reports/throughput; do [ -d "$p" ] && mkdir -p "$W/out/$p" && cp -r "$p/." "$W/out/$p/"; done
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
aq() {  # aq <gpu> <k> <port> <부하측정 0|1>: 시드 0 어댑터 병합(fp16) → AWQ → 채점
  local g=$1 k=$2 port=$3 lt=$4
  # bash는 local 한 줄의 값을 모두 먼저 펼친다. k를 쓰는 경로는 따로 선언한다(kj2·kj3에서 바깥 k가 들어감)
  local d=outputs/merged/$M/fold${k}_seed0 q=outputs/awq/$M/fold${k}_seed0
  step $g merge_f$k 600 .venv-train/bin/python -m ko_judge.merge --model $M --adapter outputs/lora/$M/fold${k}_seed0 \
    --out $d --dtype float16 || return 1
  step $g quantize_f$k 2400 .venv-train/bin/python -m ko_judge.quantize --model-dir $d --out $q --fold $k --num-samples 64 \
    || step $g quantize32_f$k 2400 .venv-train/bin/python -m ko_judge.quantize --model-dir $d --out $q --fold $k --num-samples 32 \
    || return 1
  local extra="true"
  [ "$lt" = 1 ] && extra=".venv-serve/bin/python -m ko_judge.loadtest --served $M-awq-f$k --tag lora_${M}_awq --concurrency 32 --gpu 'Tesla T4' --base-url http://localhost:$port/v1"
  step $g score_awq_f$k 4000 serve $port outputs/logs/awq_f$k $q $M-awq-f$k --dtype float16 -- bash -c "\
    .venv-serve/bin/python -m ko_judge.predict --mode score --served $M-awq-f$k --fold $k --tag lora_${M}_awq_s0 \
      --base-url http://localhost:$port/v1 && $extra"
  rm -rf $d $q
  return 0
}
gpu0() { export CUDA_VISIBLE_DEVICES=0; aq 0 0 8000 1; aq 0 2 8000 0; aq 0 4 8000 0; }
gpu1() { export CUDA_VISIBLE_DEVICES=1; aq 1 1 8001 0; aq 1 3 8001 0; }
( sleep $((LIMIT - $(date +%s))); log "HARD_STOP"; pkill -f ko_judge; pkill -f "vllm serve"; sleep 20; pack ) &
WD=$!
gpu0 & P0=$!
gpu1 & P1=$!
wait $P0; wait $P1
kill $WD 2>/dev/null
step c eval_awq 600 .venv-train/bin/python -m ko_judge.evaluate run --tag lora_${M}_awq_s0
pack
du -sh $W/out 2>/dev/null | tee -a $W/logs/steps.log
log "ALL_DONE"
'''
subprocess.run(["bash", "-c", SH])
print(open("/kaggle/working/logs/steps.log").read())
