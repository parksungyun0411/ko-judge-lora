# GPU 시간 추정 (측정 아님)

`make budget`로 생성. 프롬프트 토큰 수는 실제 데이터(2,506행)를 각 모델의 실제 토크나이저와 채팅 템플릿으로 센 값이다. 처리 속도는 가정값이라 시간은 추정이다.

설정: fold 5 × 시드 3, epoch 2.0, 행당 학습 예제 1

| 모델 | 프롬프트 토큰 합 | 평균 | p95 | 최대 | 학습 1회 토큰 | 가정 학습 속도 (tok/s) | 학습 1회 (h) |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3-4b | 5,165,208 | 2061 | 2822 | 6,874 | 6,143,254 | 2,500 | 0.68 |
| kanana-1.5-2.1b | 4,698,911 | 1875 | 2592 | 5,721 | 5,584,942 | 4,500 | 0.34 |

| 모델 | 단계 | 추정 시간 (h) |
|---|---|---:|
| qwen3-4b | train | 10.24 |
| qwen3-4b | eval_lora_bf16 | 0.44 |
| qwen3-4b | zero_shot | 0.28 |
| qwen3-4b | merge_awq | 1.25 |
| qwen3-4b | eval_merged_and_awq | 0.69 |
| qwen3-4b | loadtest | 0.20 |
| qwen3-4b | **소계** | **13.10** |
| kanana-1.5-2.1b | train | 5.17 |
| kanana-1.5-2.1b | eval_lora_bf16 | 0.31 |
| kanana-1.5-2.1b | zero_shot | 0.24 |
| kanana-1.5-2.1b | merge_awq | 1.25 |
| kanana-1.5-2.1b | eval_merged_and_awq | 0.60 |
| kanana-1.5-2.1b | loadtest | 0.20 |
| kanana-1.5-2.1b | **소계** | **7.77** |

- 두 모델 전체 (준비 1.0h, 여유 20% 포함): **약 26.3h**
- 시간이 모자랄 때 Qwen3-4B 하나만: **약 16.9h**

가정값: {"per_model": {"qwen3-4b": {"train_tps": 2500, "prefill_tps": 15000}, "kanana-1.5-2.1b": {"train_tps": 4500, "prefill_tps": 25000}}, "common": {"zero_shot_output_tokens": 300, "decode_tps_batched": 1500, "merge_awq_hours_per_fold": 0.25, "server_start_hours": 0.05, "loadtest_hours_per_variant": 0.1, "setup_hours": 1.0, "slack_ratio": 0.2}}
