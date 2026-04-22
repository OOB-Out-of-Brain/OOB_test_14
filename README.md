# Brain Stroke AI — CT 출혈 분석 파이프라인

뇌 CT 영상에서 **출혈 여부(2-class)** 를 판독하고 병변 위치를 분할(segmentation)하는 end-to-end 파이프라인입니다.

- **Classifier**: EfficientNet-B2 (2-class)
- **Segmentor**: U-Net + ResNet34 encoder (binary mask)
- **Device**: Apple Silicon MPS / NVIDIA CUDA / CPU 자동 감지
- **Pipeline 규칙 (현재 버전)**: **1% threshold** — 병변 비율 ≤ 1% 면 normal로 판독
  → normal 오탐은 적지만 작은 출혈을 놓칠 수 있음
  → 대안(분류기 단독)은 다음 레포(OOB_test_7)에 있음

---

## 1. Quick Start (5분 세팅)

```bash
# 1. 클론
git clone https://github.com/OOB-Out-of-Brain/OOB_test_6.git
cd OOB_test_6

# 2. 가상환경 + 패키지
python3 -m venv venv
source venv/bin/activate              # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. 학습용 데이터 자동 다운로드 (~3.3GB, 30분~1시간)
python scripts/download_data.py

# 4. 학습
python training/train_classifier.py
python training/train_segmentor.py

# 5. 추론
python demo.py --image path/to/ct.jpg
```

---

## 2. 데이터셋 가이드 — 뭘 받아야 하나

### 2-1. 학습용 (필수) — `python scripts/download_data.py` 한 번이면 끝

| 데이터셋 | 출처 | 용량 | 자동 여부 | 역할 |
|---|---|---|---|---|
| **tekno21** | HuggingFace `BTX24/tekno21-brain-stroke-dataset-multi` | ~560MB | ✅ 자동 | 분류 |
| **CT Hemorrhage** | PhysioNet `ct-ich v1.0.0` zip | ~1.2GB | ✅ 자동 | 분류 + 분할 |
| **AISD (synthetic)** | 로컬 생성 | ~110MB | ✅ 자동 | 분할 보조 |
| **BHSD** | HuggingFace `WuBiao/BHSD` | ~1.4GB | ✅ 자동 | 분류 + 분할 |

`download_data.py` 가 HuggingFace/PhysioNet에서 전부 가져온 뒤 BHSD 3D NIfTI를 brain window(center 40, width 80 HU)로 잘라 2D PNG로 변환.

### 2-2. 외부 테스트셋 (선택) — 학습에 **절대** 쓰지 말 것

| 데이터셋 | 출처 | 용량 | 자동 여부 | 용도 |
|---|---|---|---|---|
| **CQ500** | qure.ai | ~30GB | ❌ 이메일 등록 필요 | 외부 일반화 테스트 (FP/FN 리포트) |

CQ500 받는 법:
1. http://headctstudy.qure.ai/dataset → 이메일 등록
2. 받은 이메일 링크들(Batch zip 여러 개) 전부 다운로드
3. `data/raw/cq500/` 에 압축 해제 — 구조:
   ```
   data/raw/cq500/
     CQ500CT1/Unknown Study/*.dcm
     CQ500CT2/...
     ...
     reads.csv          (3명 방사선과의사 판독 라벨)
   ```
4. 평가: `python scripts/evaluate_cq500.py` → `results/cq500/` 에 metrics + FP/FN 샘플 생성

⚠️ CQ500은 **훈련에 사용 금지** (라이선스 NC-ND, 외부 benchmark 용도).

---

## 3. 테스트 실행 방법 (4가지)

### 3-1. 단일 이미지 추론 (가장 간단)

```bash
python demo.py --image path/to/ct.jpg
# 결과: results/{파일명}_result.png
```

### 3-2. 폴더 배치 테스트 (대량)

폴더 안의 모든 CT 이미지 자동 분류:

```bash
python scripts/run_batch_test.py \
    --input-dir /path/to/images \
    --output-dir results/my_test/
```

**출력**:
- 각 이미지의 `{이름}_result.png` (원본 | 오버레이)
- 터미널에 파일별 결과 요약 (`normal/hemorrhagic`, 신뢰도, 병변 크기)

**자동 hint**: `--input-dir` 에 normal/hemorrhagic 섞여 있어도 상관없음. 라벨 없이도 분류만 수행.

### 3-3. Val set 상세 평가 (2089장)

학습에 쓰지 않은 검증셋 전체에서 **sensitivity/specificity/FP/FN** 측정:

```bash
python scripts/evaluate_valset.py
# 출력: results/valset/metrics.txt + summary.csv + FP/FN 샘플 이미지
```

### 3-4. Pipeline 규칙 3종 비교

현재 1% threshold 규칙 vs 분류기 단독 vs 이전 세그-any 규칙 중 어느 게 최적인지 비교:

```bash
python scripts/evaluate_valset_compare.py
# 출력: results/valset/rule_comparison.txt
```

### 3-5. 외부 CQ500 테스트 (데이터 수동 다운로드 후)

```bash
python scripts/evaluate_cq500.py
# 출력: results/cq500/metrics.txt + false_positives/ + false_negatives/
```

---

## 4. Pipeline 판독 규칙 (현재: 1% threshold)

```
[이미지] → [분류기(EfficientNet-B2)] → normal/hemorrhagic 확률
         → [세그멘터(U-Net)] → 병변 픽셀 마스크

최종 판독:
  병변 비율 ≤ 1%        → normal   (미세 오탐 무시)
  병변 비율 >  1%
      & 분류기 normal   → hemorrhagic (세그가 보완)
      & 분류기 출혈     → hemorrhagic (그대로)
```

**성능 (val set 2089장 기준)**:

| 지표 | 값 |
|---|---|
| Accuracy | 72.00% |
| Sensitivity (출혈 탐지) | 28.02% ⚠️ |
| Specificity (정상 식별) | 98.09% |
| FP (오탐) | 25 |
| FN (누락) | 560 |

⚠️ 1% 규칙은 **큰 출혈 케이스(brain_test 12장)에선 100% 정확**했지만, 실제 분포(작은 출혈 포함)에선 출혈의 72%를 놓침. 스크리닝용(specificity ↑) 로만 적합.

**분류기 단독 규칙은 Acc 96.03%, Sens 93.32% 로 훨씬 균형 잡힘 → `OOB_test_7` 에 구현**.

---

## 5. 폴더 구조

```
OOB_test_6/
├─ README.md              # 이 문서
├─ HOWTRAIN.md            # 학습 상세 가이드
├─ config.yaml            # 하이퍼파라미터
├─ demo.py                # 단일 이미지 추론 데모
│
├─ data/
│  ├─ combined_dataset.py      # 분류기 로더 (CT+tekno21+BHSD)
│  ├─ ct_hemorrhage_dataset.py # 분할기 로더 (CT+BHSD 마스크)
│  └─ raw/, processed/         # .gitignore (download_data.py 로 받음)
│
├─ models/{classifier,segmentor}.py
├─ training/{train_classifier,train_segmentor,metrics}.py
├─ inference/{pipeline,visualization}.py    # 1% 규칙 적용 pipeline
│
├─ scripts/
│  ├─ download_data.py         # 🔹 학습용 4개 데이터셋 통합 다운로드
│  ├─ download_bhsd.py         # BHSD 개별
│  ├─ download_cq500.py        # 🔹 CQ500 다운로드 시도 (이메일 안내)
│  ├─ preprocess_bhsd.py       # NIfTI → 2D
│  ├─ generate_synthetic_aisd.py
│  ├─ run_batch_test.py        # 🔹 폴더 배치 테스트
│  ├─ evaluate_valset.py       # 🔹 val set 상세 평가
│  ├─ evaluate_valset_compare.py # 🔹 규칙 비교
│  └─ evaluate_cq500.py        # 🔹 외부 CQ500 평가
│
├─ checkpoints/           # 학습 결과 (gitignore)
└─ results/               # 추론 결과
```

---

## 6. 예상 소요 시간 (MacBook M 시리즈 기준)

| 작업 | 시간 |
|---|---|
| `download_data.py` (학습용 3.3GB) | 30분~1시간 |
| `train_classifier.py` (50 epoch) | ~1.5~2시간 |
| `train_segmentor.py` (early stop) | ~20~40분 |
| 단일 추론 | 1~2초 |
| Val set 전체 평가 (2089장) | ~10분 |
| CQ500 491 스캔 평가 | ~30분~1시간 |

NVIDIA GPU는 훨씬 빠름.

---

## 7. 트러블슈팅

- **PhysioNet zip 실패** → 브라우저로 https://physionet.org/content/ct-ich/1.0.0/ 수동, `data/raw/ct_hemorrhage/` 에 풀기
- **BHSD 느림** → `huggingface-cli login` 으로 인증
- **CQ500 자동 다운로드 실패** → 이메일 등록 방식이라 **정상** (수동 필요)
- **GPU OOM** → `config.yaml` 의 `batch_size` 절반 감소
- **MPS 미지원** → 자동 CPU, 느리지만 동작

---

## 8. 다음 버전

- **`OOB_test_7`** : 분류기 단독 규칙 (Acc 96%, FN 52 → 추천)
- 향후 계획: 세그멘터 recall 개선 (Dice 0.56 → 0.7+), CQ500 실측 반영
