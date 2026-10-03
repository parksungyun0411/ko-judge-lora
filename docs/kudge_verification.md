# KUDGE 확인 기록 (2026-10-03)

데이터 내용과 라이선스를 원본 파일, 저장소, 논문에서 직접 확인한 기록입니다. 수치는 `make data profile`로 다시 낼 수 있고, 결과는 [reports/dataset_profile.md](../reports/dataset_profile.md)에 있습니다.

## 출처와 고정 버전

| 출처 | 버전 | 내용 |
|---|---|---|
| GitHub `guijinSON/kudge` | 커밋 `09e13166f9f6a227cabaf82b177342a53c507d40` (2024-10-02) | `data/k2-eval-{pointwise,pairwise}[-falseinfo].csv`, 평가 코드 `src/` |
| Hugging Face `HAERAE-HUB/KUDGE` | 리비전 `7168dc844bb043fe0b0b0ee860da4dd20a67e5a2` | 같은 4개 파일(`kudge-*.csv`) + `kudge-human-annotation-raw.csv` |
| 논문 arXiv:2409.11239 | v2 (2024-10-02 갱신) | Son et al., "LLM-as-a-Judge & Reward Model: What They Can and Cannot Do" |

GitHub의 평가 파일 4개와 HF의 같은 이름 파일 4개는 sha256이 같습니다(`src/ko_judge/data.py`에 해시를 고정했습니다).

## 라이선스

| 위치 | 확인한 내용 |
|---|---|
| GitHub 저장소 | 루트 `LICENSE`가 MIT (Copyright (c) 2024 GUIJIN SON). GitHub API의 license 필드도 `mit` |
| HF 데이터셋 카드 | `license` 메타데이터 없음. 카드 본문에도 라이선스 문구 없음 |
| 논문 | arXiv 게재 라이선스는 CC BY-SA 4.0(논문 본문에 대한 것). 본문 3절은 "The dataset is made publicly available"이라 쓰고 HF 주소를 각주로 단다. 데이터 라이선스는 적혀 있지 않다 |

판단: 같은 저자가 같은 파일을 MIT로 GitHub에 배포했으므로 평가 4개 파일은 MIT로 씁니다. 그래도 저장소에는 넣지 않고 스크립트로 내려받습니다. 응답 본문은 31개 모델(GPT-4, GPT-3.5 포함)이 만든 출력이고 참조 답안도 GPT-4로 만들었다고 논문 부록 A.1에 적혀 있어, 각 모델 제공사의 이용 조건이 따로 붙을 수 있기 때문입니다. HF에만 있는 원 주석 파일은 라이선스 표기가 없어 데이터 프로파일(평가자 일치 수치 대조)에만 쓰고 학습에는 쓰지 않습니다.

## 보고된 내용 대조

| 항목 | 보고된 내용 | 확인 결과 |
|---|---|---|
| pointwise 행 | 2,506 | 2,506 (논문 Table 2, 4쪽과 같음) |
| 질문 수 | 87 | 87. 논문은 90개를 만들었다고 쓴다(3.1절). 원 주석 파일에는 90개가 있다 |
| 평가 대상 모델 | 31 | 31. 원 주석 파일에는 HyperCLOVA X를 더한 32개가 있다 |
| 점수 열 | score1/score2/final_score | 있음. final_score = (score1+score2)/2가 모든 행에서 성립 |
| pairwise 행 | 818 | 818 |
| 거짓 정보 주입 | 54행 | pointwise용 54행 + pairwise용 54행, 질문 30개. 유형 word 34 · sentence 13 · general 7 (논문 Table 7의 34/13/7과 같음, 원본의 `\x08word` 2건은 정리) |
| train 분할 | 없음 | HF 설정의 split은 모두 `test`. GitHub에도 학습용 분할 없음 |

## 논문 서술과 배포본이 다른 점

- **2점 차이 행이 남아 있습니다.** 논문 3.2절(4쪽)은 평가자 차이가 1점 이하인 경우만 평균하고 그보다 큰 경우는 뺐다고 쓰지만, 배포된 pointwise 2,506행 중 405행은 score1과 score2가 2점 다릅니다(3점 이상 차이는 없음).
- **평가자 일치율이 두 곳에서 다릅니다.** 3.2절은 "동일 또는 1점 이내 83.85%, 2점 초과 8.36%", 부록 A.2(16쪽)는 "완전 일치 52.2%, 1점 차이 30.7%, 2점 넘게 차이 17.1%"입니다. 배포된 pointwise에서는 1점 이내가 83.84%로 3.2절과 맞고, 원 주석 파일(2,757행)에서는 완전 일치 44.54%, 1점 이내 78.71%입니다.
- **우연 기준선이 다릅니다.** Table 3 캡션은 pointwise 무작위 추측 정확도를 30.86%로 적었지만, 정수 1~5 균등 추측의 기대값을 배포본으로 계산하면 27.18%입니다(`reports/dataset_profile.md`). 항상 1점을 주는 상수 판정은 43.10%입니다.
- **정확도 정의.** 논문은 "off-by-0.5 accuracy"라 부르고, 저장소 `src/pointwise_eval.py`는 `|판정 − final_score| < 1`로 계산하며 형식 실패는 분모에서 뺍니다. 판정이 정수이므로 두 정의는 같습니다.
- **pairwise.** chosen은 3.5점 이상, rejected는 "2점 이하"라고 했지만 rejected_score 2.5가 149행, 0.0·0.5가 20행 있습니다. HyperCLOVA X 응답은 pairwise에만 있습니다.
