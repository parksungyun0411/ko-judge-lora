# GPU 시간 추정 (측정 아님)

`make budget`로 생성. 프롬프트 토큰 수는 실제 데이터(2,506행)를 각 모델의 실제 토크나이저와 채팅 템플릿으로 센 값이다. 처리 속도는 가정값이라 시간은 추정이다. 속도 가정은 RTX 4090 24GB 1장 기준이며, T4처럼 bf16이 없는 GPU(fp16 경로)에는 맞지 않는다.

설정: fold 5 × 시드 3, epoch 2.0, 행당 학습 예제 1

| 모델 | 프롬프트 토큰 합 | 평균 | p95 | 최대 | 학습 1회 토큰 | 가정 학습 속도 (tok/s) | 학습 1회 (h) |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3-4b | 5,165,208 | 2061 | 2822 | 6,874 | 6,143,254 | 2,500 | 0.68 |
| kanana-1.5-2.1b | 4,698,911 | 1875 | 2592 | 5,721 | 5,584,942 | 4,500 | 0.34 |

| 모델 | 단계 | 추정 시간 (h) |
|---|---|---:|
| qwen3-4b | train | 10.24 |
| qwen3-4b | eval_lora_bf16 | 0.34 |
| qwen3-4b | zero_shot | 0.28 |
| qwen3-4b | merge_awq | 1.25 |
| qwen3-4b | eval_merged_and_awq | 0.69 |
| qwen3-4b | loadtest | 0.20 |
| qwen3-4b | **소계** | **13.00** |
| kanana-1.5-2.1b | train | 5.17 |
| kanana-1.5-2.1b | eval_lora_bf16 | 0.21 |
| kanana-1.5-2.1b | zero_shot | 0.24 |
| kanana-1.5-2.1b | merge_awq | 1.25 |
| kanana-1.5-2.1b | eval_merged_and_awq | 0.60 |
| kanana-1.5-2.1b | loadtest | 0.20 |
| kanana-1.5-2.1b | **소계** | **7.67** |

- 두 모델 전체 (준비 1.0h, 여유 20% 포함): **약 26.0h**
- 시간이 모자랄 때 Qwen3-4B 하나만: **약 16.8h**

## 축소안 (약 10h 목표, 추정)

Qwen3-4B만, 시드 0 × fold 5 (2,506행 out-of-fold) + fold 0에서 시드 1·2 (학습 7회). 병합·AWQ·부하 측정은 기본안과 같다(시드 0).

| 단계 | 추정 시간 (h) |
|---|---:|
| train | 4.78 |
| eval_lora_bf16 | 0.18 |
| zero_shot | 0.28 |
| merge_awq | 1.25 |
| eval_merged_and_awq | 0.69 |
| loadtest | 0.20 |
| **합계 (준비·여유 포함)** | **10.1** |

축소안으로 말할 수 있는 것과 없는 것:

- 가능: 시드 0 한 번의 out-of-fold 예측으로 2,506행 전체의 사람 점수 일치, 31개·보류 8개 모델 순위 τ, 질문 수별 부트스트랩 구간(질문 표본에 따른 변동), 제로샷 대 LoRA, 양자화 전후 차이, 건당 비용.
- 가능(제한적): 학습 시드에 따른 변동은 fold 0(test 질문 18개, 499행)에서 시드 3개로만 잰다.
- 불가: 전체 데이터 지표의 시드 간 구간(시드 3개 이상), 따라서 양자화 전후 차이나 제로샷 대비 개선이 학습 시드 변동보다 큰지에 대한 전체 데이터 기준 판단. 모델 간 비교(kanana 제외).

가정값: {"per_model": {"qwen3-4b": {"train_tps": 2500, "prefill_tps": 15000}, "kanana-1.5-2.1b": {"train_tps": 4500, "prefill_tps": 25000}}, "common": {"zero_shot_output_tokens": 300, "decode_tps_batched": 1500, "merge_awq_hours_per_fold": 0.25, "server_start_hours": 0.05, "loadtest_hours_per_variant": 0.1, "setup_hours": 1.0, "slack_ratio": 0.2}}
