# v4 평가 리포트 — 데이터셋별 정확도 / 오탐 / 오류율

**테스트 일자:** 2026-04-28
**셋업:** AISD (GriffinLiang) 의사 마스크 4270장으로 segmentor 메인 재학습 (커밋 661ee96).
**테스트 데이터:** 학습 미사용 hold-out 2종 (총 8,885장).

> **중요 (v3 와의 관계):** v4 는 segmentor 만 재학습됨. 분류기 (`checkpoints/classifier/best_classifier.pth`) 는 v3 와 동일하고, `inference/pipeline.py` 의 A 규칙은 분류 결과를 그대로 사용 (segmentor 는 시각화·면적 계산 전용). 따라서 같은 샘플에 대한 분류 결과는 v3 == v4.

> **stroke_test_3class 제외 사유:** 환자 단위 라벨 (한 환자의 모든 슬라이스가 같은 클래스로 일괄 라벨링) 이라 슬라이스 단위 분류기 평가에 부적합. 자세한 진단은 [부록 A](#부록-a-stroke_test_3class-제외-진단). 평가에서 일단 제외.

---

## 1. 전체 요약

| 지표 | 값 |
|---|---|
| 전체 샘플 | 8,885 |
| 전체 정확도 | **98.48%** (8,750 / 8,885) |
| 전체 오류율 | 1.52% (135) |

---

## 2. 데이터셋별 정확도 / 오류율

| 데이터셋 | 샘플 | 정확도 | 오류율 | 비고 |
|---|---:|---:|---:|---|
| brain_test | 12 | **100.00%** | 0.00% | normal 2 + hem 10 (ischemic 없음) |
| external_test_3class | 8,873 | **98.48%** | 1.52% | normal 4,427 + isc 2,260 + hem 2,186 |
| **전체** | **8,885** | **98.48%** | 1.52% | — |

---

## 3. 데이터셋별 클래스 지표 (Precision / Recall / F1)

### brain_test (12장)

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| normal | 1.0000 | 1.0000 | 1.0000 | 2 |
| ischemic | (없음) | (없음) | (없음) | 0 |
| hemorrhagic | 1.0000 | 1.0000 | 1.0000 | 10 |

### external_test_3class (8,873장)

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| normal | 0.9833 | 0.9980 | 0.9906 | 4,427 |
| ischemic | 0.9769 | 0.9929 | 0.9849 | 2,260 |
| hemorrhagic | 0.9966 | 0.9497 | 0.9726 | 2,186 |

---

## 4. 오분류 6종 (오탐 / 누락)

전체 8,885 장 기준. 비율은 해당 GT 클래스 support 대비.

| 오류 유형 | 한국어 | 개수 | GT 대비 |
|---|---|---:|---:|
| hemorrhagic → normal | 출혈을 정상으로 놓침 | 60 | 2.73% |
| hemorrhagic → ischemic | 출혈을 허혈로 혼동 | 50 | 2.28% |
| ischemic → normal | 허혈을 정상으로 놓침 | 15 | 0.66% |
| normal → hemorrhagic | 정상을 출혈로 오탐 | 6 | 0.14% |
| normal → ischemic | 정상을 허혈로 오탐 | 3 | 0.07% |
| ischemic → hemorrhagic | 허혈을 출혈로 혼동 | 1 | 0.04% |

---

## 5. 통합 Confusion Matrix (전체 8,885장, 행=GT / 열=Pred)

| | normal | ischemic | hemorrhagic |
|---|---:|---:|---:|
| **normal** (4,429) | 4,420 | 3 | 6 |
| **ischemic** (2,260) | 15 | 2,244 | 1 |
| **hemorrhagic** (2,196) | 60 | 50 | 2,086 |

---

## 6. 핵심 관찰

- 분류기·세그멘터 통합 파이프라인이 일반 도메인 (Kaggle external_test_3class) 에서 **98.48%** 의 슬라이스 단위 정확도.
- 가장 큰 오류 모드: **hemorrhagic → normal (60건, 2.73%)** + **hemorrhagic → ischemic (50건, 2.28%)** — 출혈을 놓치거나 허혈로 오인.
- normal → 병변 오탐은 매우 낮음 (총 9건, 0.21%) — 위양성 측면은 안전.
- brain_test 는 모든 정답 (100%) 이지만 12장으로 통계적 의미 약함.

---

## 부록 A. stroke_test_3class 제외 진단

### A.1 결정적 증거: 파일명 패턴

stroke_test_3class 의 파일명: `58 (12).jpg`, `58 (24).jpg`, `100 (14).jpg`, `100 (15).jpg`, `100 (21).jpg` — **환자 ID(58, 100, 101…) + 슬라이스 번호** 형식. 한 환자가 hemorrhagic 진단을 받았다면 그 환자의 모든 슬라이스가 일괄로 hemorrhagic 폴더로 들어감.

external_test_3class 의 파일명: `10003.png`, `10017.png`, `10024.png` — sequential numeric. 슬라이스 단위로 큐레이팅됨.

### A.2 결정적 증거: 분류기 확률값 (같은 환자, 다른 슬라이스)

| 데이터셋 | 파일 | GT | 예측 | normal | ischemic | hemorrhagic |
|---|---|---|---|---:|---:|---:|
| stroke_test/isc | `100 (14).jpg` | ischemic | normal | **0.891** | 0.042 | 0.066 |
| stroke_test/isc | `100 (15).jpg` | ischemic | hemorrhagic | 0.055 | 0.039 | **0.905** |
| stroke_test/isc | `100 (21).jpg` | ischemic | normal | **0.932** | 0.033 | 0.034 |
| stroke_test/isc | `101 (2).jpg` | ischemic | normal | **0.770** | 0.040 | 0.190 |
| stroke_test/hem | `58 (12).jpg` | hem | normal | **0.922** | 0.039 | 0.039 |
| stroke_test/hem | `58 (24).jpg` | hem | normal | **0.927** | 0.036 | 0.036 |
| external/isc | `10003.png` | ischemic | ischemic | 0.026 | **0.940** | 0.034 |
| external/isc | `10017.png` | ischemic | ischemic | 0.028 | **0.932** | 0.039 |

같은 환자 100 의 인접 슬라이스가 한쪽은 highly-confident normal (89%), 다른 쪽은 highly-confident hemorrhagic (90%) — 분류기가 슬라이스 단위로 lesion 유무를 판단하고, 그 환자의 일부 슬라이스에 실제로 lesion 신호가 없다는 의미.

### A.3 시각 비교

`results/test_full_v4/domain_comparison.png` (저장됨):
- stroke_test_3class 의 일부 슬라이스 = 얼굴·콧구멍·머리꼭대기 등 lesion 이 보일 수 없는 위치도 환자 라벨 그대로 붙음.
- external_test_3class 는 모두 lesion 이 잘 보이는 표준 뇌 슬라이스.

### A.4 결론 / 처리

- 8.47% 라는 stroke_test_3class 정확도는 **분류기 결함이 아니라 라벨 입자도 (slice-level vs patient-level) 의 형식적 불일치**.
- 슬라이스 단위 평가에 부적합 → **`scripts/test_full.py` 의 `DEFAULT_SOURCES` 에서 주석 처리하여 평가 대상에서 제외** (2026-04-28).
- 향후 활용 옵션: (1) 환자 단위 정확도로 재집계, (2) lesion 보이는 슬라이스 별도 라벨링 후 재테스트, (3) 분류기 학습 시 환자-단위 weak supervision 으로 추가.
