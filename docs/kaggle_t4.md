# Kaggle T4 실행 기록

2026-10-16부터 10-26까지 Kaggle 무료 GPU(T4 두 장)로 GPU 단계를 돌린 절차와 결과다.
원래 계획은 RTX 4090 임대([runpod_setup.md](runpod_setup.md))였지만, 비용 없이 돌릴 수 있는 Kaggle T4로 바꿨다.
판정 지표는 `reports/metrics/`, 결과 표는 [reports/RESULTS.md](../reports/RESULTS.md)에서 나온다.
그 밖의 비교 수치(예측끼리의 일치율, AWQ 점수 차이, 학습 손실·시간, vLLM 처리량·메모리)는 `make checks`가 저장소 안의 예측·로그에서 다시 내는
[reports/run_checks.json](../reports/run_checks.json)에서 나온다. 예측(`outputs/predictions/`)에는 점수와 점수 분포만 있고 응답 본문은 없다.
커널 스크립트, 단계 로그, GPU 정보(`nvidia-smi`), vLLM 로그 발췌, AWQ 메모리 부족 오류 줄은 [kaggle_t4/](kaggle_t4/)에 있다.
이 실행에서 잰 값 가운데 저장소에 근거 파일이 없는 것은 kj-smoke 점검값과 토크나이저로 다시 센 토큰 수이고, 해당 문단에 그렇게 적었다.

## 환경

| 항목 | 값 |
|---|---|
| 실행 | Kaggle 스크립트 커널, `kaggle kernels push --accelerator NvidiaTeslaT4`, 비공개 |
| GPU | Tesla T4 15GB × 2 (compute capability 7.5, bf16 없음), 드라이버 580.178.04 ([kaggle_t4/gpu_kj*.txt](kaggle_t4/), 네 커널 모두 같음) |
| 한도 | 주 30시간, 커널 하나 최대 12시간. GPU 커널 두 개를 동시에 돌릴 수 있었다 |
| 판정 모델 | kanana-1.5-2.1b (`kakaocorp/kanana-1.5-2.1b-instruct-2505`, 커밋 `7df4bc3`) |

커널은 GitHub에서 고정 커밋(`a2b5d90`)을 받아 `uv`로 학습 환경(`.venv-train`, [requirements/train.txt](../requirements/train.txt))과
서빙 환경(`.venv-serve`, [requirements/serve.txt](../requirements/serve.txt))을 만든 뒤 돌린다. KUDGE는 커널 안에서 `make data`와 같은 경로로 내려받는다.
T4에는 bf16이 없어 학습은 [configs/t4/lora_kanana-1.5-2.1b.yaml](../configs/t4/lora_kanana-1.5-2.1b.yaml)(정밀도만 fp16)로, 서빙은 `--dtype float16`으로 했다.
커널 스크립트는 실행한 그대로 [kaggle_t4/kj1.py](kaggle_t4/kj1.py)~[kj4.py](kaggle_t4/kj4.py)에 있다. `make` 타깃 대신 같은 모듈(`ko_judge.train_lora`, `predict`, `merge`, `quantize`, `evaluate`, `loadtest`)을 직접 불렀다.

```bash
kaggle kernels push -p kj1/ --accelerator NvidiaTeslaT4
kaggle kernels status <id>              # RUNNING → COMPLETE
kaggle kernels output <id> -p out/kj1   # 끝난 뒤에만 받을 수 있다
```

한 커널 안에서 GPU 두 장을 나눠 썼다. 작업마다 `CUDA_VISIBLE_DEVICES`로 한 장씩 주고, vLLM 서버를 동시에 띄울 때는 `PORT`로 포트를 나눴다
(판정·부하 측정에는 같은 포트를 `--base-url`로 넘겼다. 지금은 `predict`·`loadtest`가 `PORT`를 기본값으로 읽는다).
뒤 커널은 앞 커널의 출력(어댑터 tgz)을 `kernel_sources`로 붙여 받았다. 커널마다 11시간 20분 한도를 두고,
단계를 시작하기 전에 남은 시간이 그 단계의 예상 시간보다 짧으면 `SKIP`을 남기고 건너뛰게 했다.

## 실행 순서와 소요 시간

커널별 시간은 단계 로그의 `SETUP_DONE`부터 `ALL_DONE`까지(UTC)이고, 그 앞의 환경 준비(저장소 받기, uv 환경 두 개, 데이터·모델 내려받기)는 들어 있지 않다.

### kj-smoke (10-16)

T4에 들어가는지와 속도만 봤다. 이 커널의 로그는 저장소에 넣지 않았고, 아래 값은 그 로그에서 옮겼다(일부는 kj1.py 머리 주석에도 있다).

| 확인 | 결과 |
|---|---|
| 최장 프롬프트 배치 2 학습 | 최대 8.3GiB, T4에 들어감 |
| 학습 속도 | 55 s/스텝 → fold 하나 약 2.9시간 |
| 제로샷 판정 fold 0 (511건) | 42분 (재시도 197건) |
| 점수 모드 | `int('①')` 오류 → 파싱을 ASCII 숫자만 받도록 고침(`a2b5d90`) |

### kj-k1 (10-16 04:02 → 14:24, 10시간 22분)

| GPU | 단계 | 소요 |
|---|---|---:|
| 0 | 시드 0 fold 0 → 2 → 4: 학습 → 어댑터 서빙 점수 판정 | fold당 학습 2시간 59분~3시간 1분, 판정 17~19분 |
| 1 | 제로샷 2,506행 (생성형, 논문 프롬프트) | 3시간 39분 |
| 1 | 시드 0 fold 1 → 3 | 학습 2시간 57분·3시간 2분, 판정 21분·17분 |

### kj-k2 (10-23 00:51 → 09:41, 8시간 50분)·kj-k3 (02:14 → 11:28, 9시간 14분)

두 커널을 동시에 돌렸다. kj-k2는 시드 1, kj-k3는 시드 2의 다섯 fold를 학습·판정했다.
남는 GPU 시간에 시드 0 어댑터를 병합(fp16)해 fold별로 서빙·판정했고(kj-k2 fold 0·1, kj-k3 fold 2·3·4), fold 0에서 부하를 쟀다.
두 커널 모두 건너뛴 단계는 없었다.

| 단계 | 소요 |
|---|---:|
| 학습 (fold당 1,464~1,492행, 2 epoch) | 2시간 35분~2시간 51분 |
| 어댑터 서빙 점수 판정 (fold당 497~539건) | 16~21분 |
| 병합 | 1분 |
| 병합 모델 판정 | 11~14분 (fold 0은 부하 측정 1,000건 포함 38분) |
| AWQ 양자화 | 다섯 fold 모두 30초 만에 실패 (아래 문제 1) |

열다섯 번 학습한 `train_runtime`은 2.57~3.03시간, 학습 손실은 0.570~0.671이었다
(`outputs/lora/kanana-1.5-2.1b/fold*_seed*/train_summary.json`, 가중치는 저장소에 넣지 않음). 단계별 소요는 [kaggle_t4/steps_kj*.log](kaggle_t4/)의 시각에서 셌다.

### kj-k4 (10-26 00:41 → 02:41, 2시간 0분)

AWQ만 다시 돌렸다. kj-k1 출력의 시드 0 어댑터를 fold마다 병합 → AWQ → 서빙·판정했고(GPU 0은 fold 0·2·4, GPU 1은 fold 1·3), fold 0에서 부하를 쟀다.
보정 샘플 64개로 먼저 시도하고, 실패하면 32개로 다시 돌리는 단계를 미리 넣어 두었다.

| 단계 | 소요 |
|---|---:|
| 병합 | 1분 안팎 |
| AWQ, 보정 64개 | 다섯 fold 모두 20여 초 만에 OOM |
| AWQ, 보정 32개 | 다섯 fold 모두 성공, 15~17분 |
| AWQ 모델 판정 | 13~15분 (fold 0은 부하 측정 1,000건 포함 39분) |

## 문제와 해결

### 1. AWQ 보정이 T4 메모리를 넘음

kj-k2·kj-k3의 양자화는 다섯 fold 모두 보정 단계에서 `torch.OutOfMemoryError`로 멈췄다(오류 줄은 [kaggle_t4/quantize_oom.log](kaggle_t4/quantize_oom.log)). 보정 샘플 256개를 한 번에 캐시하는데,
판정 프롬프트가 길어(kanana 토크나이저로 중앙값 1,901토큰, 90 백분위 2,358, 최대 5,723) 15GB를 넘었다.
길이를 2,048로 자르면 샘플의 35%에서 끝의 정답 점수가 잘리므로, 길이 상한(4,096, 잘리는 샘플 0.5%)은 두고 샘플 수를 64개로 줄였다.
64개(fold별 12.1만~12.8만 토큰)도 kj-k4에서 다섯 fold 모두 OOM이 났고, 32개(6.2만~6.8만 토큰)에서 성공했다.
32개의 토큰 수는 AWQ 저자 구현([mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq) `entry.py`)의 보정량 128개 × 512토큰(65,536)과 비슷하다.
저장소의 `make merge-quant`에는 `QUANT_ARGS="--num-samples 32"`로 넘긴다. 이 문단의 토큰 수는 `quantize.py`와 같은 표본 추출(시드 0)을 kanana 토크나이저로 로컬에서 다시 센 값이고, 센 결과 파일은 저장소에 없다.

AWQ 모델은 병합 fp16보다 점수를 낮게 줬다. 같은 fold의 병합 모델과 정수 점수가 다른 행이 2,506행 중 707행이고, 그중 700행에서 AWQ 쪽이 낮았다
(평균 점수 2.04 → 1.74, 사람 평균 2.15). fold별 일치율(거짓 정보 응답 포함)은 54~86%로 fold마다 차이가 컸다. 31개 모델 가운데 29개의 평균 점수가 내려갔고(모델별 0~0.64점), 모델 순위 τ는 0.865에서 0.863으로 거의 그대로였다.
off-by-0.5 정확도는 0.679에서 0.660으로 내려갔다. 점수 차이 수치는 run_checks.json의 `awq_vs_merged`·`awq_vs_merged_by_fold`에 있다.

### 2. 커널 스크립트의 병합 폴더 이름

kj-k2·kj-k3 커널 스크립트([kaggle_t4/kj2.py](kaggle_t4/kj2.py)의 `mq`)의 `local g=$1 k=$2 ... d=outputs/merged/$M/fold${k}_seed0` 한 줄에서, bash는 값을 모두 먼저 펼치므로
`${k}`에 바깥 반복문의 마지막 값이 들어갔다. 그래서 병합 모델이 fold 번호와 다른 이름의 폴더에 저장됐다.
fold k 어댑터를 병합한 직후 그 폴더를 서빙해 판정했으므로 판정한 모델은 맞았다. 이를 예측으로 확인했다.
같은 시드·같은 fold의 어댑터 서빙 판정과 병합 모델 판정의 정수 점수가 fold마다 99.6~100% 같았다(fold당 497~539건, 거짓 정보 응답 포함).
같은 fold라도 시드가 다른 어댑터끼리는 68~82%만 같다(run_checks.json의 `adapter_s0_vs_merged`·`adapter_seed_pairs`). 다음 커널(kj-k4)에서는 경로를 따로 선언했다.

### 3. T4에서 판정 처리량은 프롬프트 처리 쪽에서 막힌 것으로 보인다

병합 fp16 모델의 부하 측정(동시 32, 1,000건)은 0.73건/s, p95 52.7초였다. 점수 판정은 출력이 1토큰이라 처리 시간 대부분이 프롬프트 처리(prefill)일 것으로 본다. prefill과 decode 시간을 따로 재지는 않았다.
이 서버의 vLLM 로그에서 10초 간격 프롬프트 처리량(점수 판정과 부하 측정 구간, 0이 아닌 210개)은 중앙값 초당 약 1,250토큰(10~90 백분위 약 1,080~1,460)이었고,
그중 205개 구간에서 한 번에 2~3건만 돌고 나머지는 대기했다. T4에서 vLLM 0.30.0은 TRITON_ATTN 백엔드를 썼다.
로그 발췌는 [kaggle_t4/vllm_merged_f0.log](kaggle_t4/vllm_merged_f0.log)·[vllm_awq_f0.log](kaggle_t4/vllm_awq_f0.log), 수치는 run_checks.json의 `vllm_f0`에 있다.

AWQ 모델(vLLM이 `MarlinLinearKernel`로 실행)은 가중치 메모리가 3.95GiB에서 1.34GiB로 줄어 KV 캐시가 62,736토큰에서 87,296토큰으로 늘었다.
그래도 대부분의 구간(215개 중 201개)에서 한 번에 2~3건만 돌았고 부하 측정은 0.71건/s, p95 54.1초로 병합 fp16(0.73건/s)과 비슷했다(프롬프트 처리량 중앙값 초당 약 1,230토큰). 메모리를 줄여도 이 작업의 처리량은 늘지 않았다.
prefill 병목 말고 4bit 커널의 부담이 메모리 이득을 상쇄했을 가능성도 따로 확인하지 않았다.
서빙 설정은 최대 길이·메모리 비율·dtype 말고는 vLLM 기본값으로 두었다.

## 측정 조건

- 판정 지표는 out-of-fold 예측 2,506행 전체로 낸다. 보류 모델 8개(654행)는 질문과 모델을 모두 처음 보는 상태다.
- 학습 시드 3개(0·1·2) 모두 다섯 fold를 끝냈다. 결과 표의 시드 묶음 칸은 평균(최소~최대)이다.
- 병합·AWQ는 시드 0만 돌렸다. AWQ 보정 샘플은 32개다.
- 부하 측정은 fold 0 모델, 동시 요청 32, 판정 1,000건(워밍업 제외)이다. 측정 중 같은 커널의 다른 GPU에서는 병합 모델 측정 때 학습이, AWQ 측정 때 양자화·판정이 돌았다.
- `$/100만 건`은 T4 시간당 $0.1069(Colab 종량제 단가 × T4 소비율)로 환산했다. 측정 자체는 무료 Kaggle에서 했다.
- Qwen3-4B는 돌리지 않았다. 사유는 결과 표 아래에 있다.
