# reports

이 폴더의 파일은 모두 스크립트가 만듭니다. 손으로 고치지 않습니다.

| 경로 | 만드는 명령 | 상태 |
|---|---|---|
| `dataset_profile.{md,json}` | `make data profile` | 생성됨 (실제 데이터, 모델 실행 없음) |
| `gpu_budget.{md,json}` | `make budget` | 생성됨 (토큰 수는 실측, 시간은 추정) |
| `metrics/{tag}.json` | `make zeroshot·eval-lora·eval-merged·eval-awq` | 측정 전 |
| `throughput/{tag}.json` | `make eval-merged·eval-awq` | 측정 전 |
| `RESULTS.md` | `make report` | 표만 있음, 모든 칸 측정 전 |
