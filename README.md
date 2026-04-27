# Brain Stroke AI — CT 뇌졸중 3-class 분석

뇌 CT 이미지를 받아 **정상 / 허혈성(ischemic) / 출혈성(hemorrhagic)** 판독과 병변 위치 시각화를 수행.

- **Classifier**: EfficientNet-B2 (3-class)
- **Segmentor**: U-Net (ResNet34/EfficientNet-B0 encoder, 3채널 softmax — bg/ischemic/hemorrhagic)
- **Device**: Apple Silicon MPS / NVIDIA CUDA / CPU 자동
- **판독 규칙**: 분류기 softmax 결과를 그대로 사용 (post-processing 없음). 세그멘터 마스크는 위치 시각화 전용.

---

## 1. 빠른 시작 (clone → 학습 → 평가)

```bash
git clone https://github.com/OOB-Out-of-Brain/OOB_test_12.git
cd OOB_test_12

python3 -m venv venv
source venv/bin/activate                  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 학습 (필요한 데이터는 학습 시작 시 자동 다운로드됩니다)
python training/train_classifier.py --epochs 50
python training/train_segmentor.py  --epochs 80

# 외부 테스트셋(학습 비사용 hold-out) 평가
python scripts/evaluate_external_test.py
```

체크포인트(.pth)는 용량이 커서 레포에 포함되지 않습니다. 학습이 끝나면 각각
`checkpoints/classifier/best_classifier.pth`, `checkpoints/segmentor/best_segmentor.pth` 에 저장됩니다.

---

## 2. 데이터셋

### 2-1. 학습용 (자동 다운로드)

학습 스크립트가 진입 시점에 누락 데이터를 점검하고 자동으로 받습니다. 사전 setup 단계가 따로 필요 없습니다.

| 데이터셋 | 출처 | 용량 | 인증 | 역할 | 3-class 매핑 |
|---|---|---|---|---|---|
| **tekno21** | HuggingFace `BTX24/tekno21-brain-stroke-dataset-multi` | ~560MB | 없음 | 분류 | Kanama→hem, iskemi→isc, İnme Yok→normal |
| **BHSD** | HuggingFace `WuBiao/BHSD` | ~1.4GB | 없음 | 분류 + 분할 (출혈) | 5 subtype 통합 → hemorrhagic |
| **CPAISD** | Zenodo 10892316 (CC-BY-4.0) | ~5.6GB | 없음 | 분류 + 분할 (허혈) | core+penumbra → ischemic |

**3-class 균형** (자동 다운로드만으로):
- normal:       tekno21 İnme Yok
- ischemic:     tekno21 iskemi + CPAISD ~10,000 슬라이스 (실제 NCCT, 의사 마스크)
- hemorrhagic:  tekno21 Kanama + BHSD ~5,000 슬라이스 (NIfTI, brain window 처리)

WeightedRandomSampler 가 분류 학습 시 클래스 불균형을 자동 보정합니다.

### 2-2. 옵션 (인증/특수 사정)

| 데이터셋 | 출처 | 용량 | 사용 | 비고 |
|---|---|---|---|---|
| CT Hemorrhage | PhysioNet `ct-ich v1.0.0` | ~1.2GB | `--with-ct` | 인증 필요 (PHYSIONET_USER/PASS 또는 Kaggle 미러) |
| 합성 AISD | 로컬 생성 | ~110MB | `--with-synthetic-aisd` (세그만) | CPAISD 가 진짜 데이터를 제공하므로 비권장 |
| CQ500 | Academic Torrents | ~28GB | `python scripts/download_cq500.py` | 외부 평가 (출혈 vs 비출혈 이진) |

### 2-3. 외부 테스트셋 (학습에 안 씀, 자동 다운로드)

**`ozguraslank/brain-stroke-ct-dataset`** (Kaggle, 3-class)
- 자동 다운로드: `python scripts/evaluate_external_test.py` 가 알아서 받음 (Kaggle CLI 필요)
- 학습 데이터(tekno21/BHSD/CPAISD) 와 무관 → 진짜 일반화 평가
- 마스크 없이 클래스 라벨만 — 분류기 평가 전용

**Kaggle CLI 셋업 (1회):**
```bash
pip install kaggle
# Kaggle → Account → Create New API Token → ~/.kaggle/kaggle.json 배치
chmod 600 ~/.kaggle/kaggle.json
```

---

## 3. 학습

### 3-1. 분류기

```bash
python training/train_classifier.py --epochs 50                       # 기본: tekno21 + BHSD + CPAISD
python training/train_classifier.py --epochs 50 --batch_size 8        # 느린 머신
python training/train_classifier.py --tekno21-only                    # 다른 데이터 다 빼고 tekno21 만
python training/train_classifier.py --no-cpaisd                       # CPAISD 빼고 학습
python training/train_classifier.py --with-ct                         # CT Hemorrhage 추가 (PhysioNet 인증)
```
체크포인트: `checkpoints/classifier/best_classifier.pth`

### 3-2. 세그멘터

```bash
python training/train_segmentor.py --epochs 80                        # 기본: BHSD(hem) + CPAISD(isc) + tekno21 pseudo
python training/train_segmentor.py --epochs 80 --encoder resnet34     # encoder 교체
python training/train_segmentor.py --no-cpaisd                        # CPAISD 빼고 (테스트용)
python training/train_segmentor.py --with-synthetic-aisd              # 합성 AISD 도 추가
python training/train_segmentor.py --with-ct                          # CT Hemorrhage 추가
```
체크포인트: `checkpoints/segmentor/best_segmentor.pth`

### 3-3. 실시간 학습 모니터링 (별도 터미널)

```bash
python scripts/watch_training.py            # logs/ 최신 .log 자동
python scripts/watch_training.py --no-bar   # 진행바 감추고 epoch 결과만
```

### 3-4. 예상 소요 (MacBook M 시리즈, MPS)

| 작업 | 소요 시간 |
|---|---|
| 첫 실행시 데이터 자동 다운로드 (BHSD ~1.4GB + CPAISD ~5.6GB) | 30분 ~ 1시간 |
| 분류기 50 epoch | ≈ 1시간 30분 ~ 2시간 |
| 세그멘터 80 epoch | 1 ~ 1.5시간 |
| 외부 테스트셋 평가 | 5 ~ 10분 |

---

## 4. 추론 / 평가

### 4-1. 단일 이미지

```bash
python demo.py --image path/to/ct.png
# → results/{파일명}_result.png  (원본 + 3-class 확률 바 + 병변 overlay)
```

### 4-2. 폴더 배치

```bash
python scripts/run_batch_test.py --input-dir /path/to/imgs --output-dir results/my_run
python scripts/run_batch_test.py --input-dir ... --output-dir ... --no-gt-from-name
```
파일명에 `nomal/normal`, `iskemi/ischem`, `EDH/ICH/SAH/SDH/hemorr` 등이 있으면 GT 자동 인식.

### 4-3. 외부 테스트셋 평가 (가장 중요 — 실제 일반화 성능)

```bash
python scripts/evaluate_external_test.py
# → results/external_3class/
#     ├─ metrics.txt    (3×3 confusion + per-class precision/recall/F1)
#     ├─ summary.csv    (샘플별 GT/예측/확률)
#     ├─ correct/{class}/
#     └─ wrong/{gt}_to_{pred}/   (오분류 9 버킷)
```

이 데이터셋은 학습에 절대 사용되지 않은 외부 hold-out 입니다. 진짜 일반화 성능 측정이 가능합니다.

### 4-4. 내부 검증 평가

```bash
python scripts/evaluate_valset.py        # 학습 시 환자 단위로 분리해둔 val (≈2361장)
python scripts/evaluate_ischemic.py      # tekno21 iskemi val 서브셋만
python scripts/evaluate_cq500.py         # CQ500 외부 (출혈 이진 평가, 28GB 다운로드 필요)
```

---

## 5. 파이프라인 판독 규칙

```
[이미지] → [분류기]   → softmax(3) → 최종 판독 (argmax)
        ↓ (선택, 시각화용)
        [세그멘터] → 픽셀별 (bg/isc/hem) → overlay
                   ischemic=파란톤, hemorrhagic=빨간톤
```

- 분류기 결과 그대로 신뢰. 1% threshold 같은 후처리 없음.
- 세그멘터는 "어디에 병변인가?" 시각화 용도.

---

## 6. 폴더 구조

```
OOB_test_12/
├─ README.md
├─ config.yaml                      # 하이퍼파라미터, 데이터 경로
├─ requirements.txt
├─ demo.py                          # 단일 이미지 추론
│
├─ data/
│  ├─ combined_dataset.py           # 분류기 3-class 통합 로더 (tekno21 + BHSD + CPAISD)
│  ├─ seg_dataset.py                # 세그멘터 3-class 로더 (BHSD + CPAISD + 옵션들)
│  ├─ auto_prepare.py               # 학습 시 누락 데이터 자동 다운로드 헬퍼
│  ├─ raw/        (gitignore)
│  └─ processed/  (gitignore)
│
├─ models/
│  ├─ classifier.py                 # EfficientNet-B2
│  └─ segmentor.py                  # U-Net 3-channel softmax
│
├─ training/
│  ├─ train_classifier.py           # 3-class 분류 학습
│  ├─ train_segmentor.py            # 3-class 세그 학습
│  └─ metrics.py
│
├─ inference/
│  ├─ pipeline.py                   # StrokePipeline (분류 → 세그 → overlay)
│  └─ visualization.py
│
├─ scripts/
│  ├─ download_cpaisd.py            # CPAISD (Zenodo 자동, 인증 불필요)
│  ├─ preprocess_cpaisd.py          # CPAISD .npz → 2D PNG + brain window
│  ├─ download_bhsd.py
│  ├─ preprocess_bhsd.py            # BHSD NIfTI → 2D PNG
│  ├─ download_data.py              # tekno21 + BHSD + (옵션) CT Hem 통합 진입점
│  ├─ generate_synthetic_aisd.py    # 합성 AISD (옵션, 백업용)
│  ├─ generate_ischemic_pseudo_masks.py  # tekno21 Grad-CAM pseudo (분류기 학습 후)
│  ├─ download_external_test.py     # ⭐ 외부 3-class 테스트셋 (Kaggle)
│  ├─ evaluate_external_test.py     # ⭐ 외부 일반화 평가
│  ├─ evaluate_valset.py            # 내부 val 평가
│  ├─ evaluate_ischemic.py          # tekno21 iskemi val 서브셋
│  ├─ download_cq500.py / evaluate_cq500.py   # CQ500 외부 (옵션 28GB)
│  ├─ run_batch_test.py             # 폴더 배치 추론
│  ├─ setup_all.py                  # (옵션) 사전 셋업 — 학습 스크립트가 자동 호출하므로 보통 불필요
│  ├─ watch_training.py
│  └─ _eval_common.py
│
├─ checkpoints/   (gitignore — 학습 결과)
├─ logs/          (gitignore — 학습 로그)
└─ results/       (gitignore — 추론 결과)
```

---

## 7. 데이터 누수 (data leakage) 방지 정책

| 평가 종류 | 데이터 출처 | 학습 데이터와 겹침? |
|---|---|---|
| `evaluate_valset.py` | tekno21/BHSD/CPAISD val split (환자 단위) | 환자 단위로 학습-검증 분리됨 → 누수 없음 |
| `evaluate_ischemic.py` | tekno21 iskemi val | 분류기 학습엔 안 씀 → 누수 없음 |
| ⭐ `evaluate_external_test.py` | **Kaggle ozguraslank/brain-stroke-ct-dataset** | **완전히 다른 출처** → 진짜 일반화 평가 |
| `evaluate_cq500.py` | CQ500 (qure.ai) | 학습 미사용. ICH 라벨만이라 출혈 이진 평가 |

배포/논문 발표용 성능 지표는 ⭐ 외부 테스트셋 결과로 보고하는 것이 옳습니다.

---

## 8. 트러블슈팅

- **CPAISD 다운로드 5.6GB가 너무 큼** → 첫 실행 1회만 받으면 끝. Zenodo 직링크라 인증 불필요.
- **CPAISD `.npz` 형식 오류** → [scripts/preprocess_cpaisd.py](scripts/preprocess_cpaisd.py) 의 `_pick_2d_array`/`_to_uint8_image`/`_to_binary_mask` 함수가 다양한 형식을 자동 탐지. 슬라이스 0장이 나오면 그쪽 1-2줄 조정.
- **외부 테스트셋 다운로드 실패 (Kaggle)** → `pip install kaggle` + `~/.kaggle/kaggle.json` 배치 + `chmod 600`.
- **CT Hemorrhage 401/403** → 옵션 데이터입니다. 안 받아도 학습 정상 진행.
- **GPU OOM** → `config.yaml` 의 `batch_size` 절반.
- **MPS 미지원** → 자동 CPU 전환.
- **체크포인트 없음 에러** → `python training/train_classifier.py --epochs 50` 먼저.
- **세그멘터 ckpt 없이 평가 돌림** → 분류만 수행, overlay 생략 (정상 동작).
- **학습 중 파일 rename 금지** → DataLoader worker 가 경로를 못 찾아 죽음.

---

## 9. 라이선스

- 코드: 프로젝트 정책에 따름
- CPAISD: CC-BY-4.0 (Zenodo 10892316)
- BHSD: NC-ND
- tekno21: HuggingFace 데이터셋 정책
- CQ500: CC BY-NC-SA 4.0 (학습 금지, 평가 전용)
- 외부 테스트셋(Kaggle): 각 데이터셋 페이지 라이선스 확인
