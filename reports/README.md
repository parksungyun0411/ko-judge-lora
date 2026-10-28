# reports

이 폴더의 파일은 모두 스크립트가 만듭니다. 손으로 고치지 않습니다.

| 경로 | 만드는 명령 | 상태 |
|---|---|---|
| `dataset_profile.{md,json}` | `make data profile` | 생성됨 (실제 데이터, 모델 실행 없음) |
| `gpu_budget.{md,json}` | `make budget` | 생성됨 (토큰 수는 실측, 시간은 추정) |
| `metrics/{tag}.json` | `make zeroshot·eval-lora·eval-merged·eval-awq` | kanana-1.5-2.1b 측정 완료 (Kaggle T4, 2026-10-16~26). `_fold{k}`는 커널 안에서 fold 하나씩 낸 중간 지표. 커널은 같은 `python -m ko_judge.evaluate run`을 직접 불렀고, 저장소의 `outputs/predictions/{tag}/`로 다시 내면 같은 값이 나온다(`tests/test_run_checks.py`) |
| `throughput/{tag}.json` | `make eval-merged·eval-awq` | kanana-1.5-2.1b 병합·AWQ, fold 0, 동시 32 |
| `RESULTS.md` | `make report GPU_NAME="Tesla T4" GPU_USD_PER_HOUR=0.1069 PRICE_NOTE=…` | 생성됨. Qwen3-4B 행은 측정 안 함(사유 포함) |
| `run_checks.json` | `make checks` | 생성됨. 결과 표 밖의 비교 수치(어댑터 대 병합, 시드 간, AWQ 대 병합 점수 차이, 학습 손실·시간, vLLM 로그의 처리량·메모리). 입력은 `outputs/predictions`, `outputs/lora/*/*/train_summary.json`, `docs/kaggle_t4/` |
