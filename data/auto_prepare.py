"""학습 시 데이터셋 누락 자동 보충.

분류기/세그멘터 어느 쪽이든 학습 시작 전에 `ensure_training_data()` 한 번만 부르면
필요한 모든 원본/전처리물이 준비된다. 이미 있으면 즉시 반환 (idempotent).

자동 준비 대상:
  - CT Hemorrhage (PhysioNet 또는 Kaggle 미러) — download_data.py 가 처리
  - tekno21        (HuggingFace, 자동)         — load_dataset() 가 처리
  - BHSD           (HuggingFace + 2D 슬라이싱) — download_bhsd.py + preprocess_bhsd.py
  - 합성 AISD      (로컬 생성)                  — generate_synthetic_aisd.py
  - CPAISD         (Zenodo)                    — download_cpaisd.py + preprocess_cpaisd.py

CPAISD 는 별도 트리거가 있지만 (seg_dataset / combined_dataset 내부에서 자동 호출),
여기서도 한 번 미리 부르면 학습 진입 직후 모든 다운로드가 끝나고 시작된다.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"

# 누락 판정용 sentinel 경로 (가장 안쪽 파일을 본다)
CT_HEM_SENTINEL = REPO_ROOT / "data" / "raw" / "ct_hemorrhage" / \
    "computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.0.0" / \
    "hemorrhage_diagnosis.csv"
BHSD_PROCESSED_SENTINEL = REPO_ROOT / "data" / "processed" / "bhsd" / "index.csv"
AISD_SYNTH_SENTINEL = REPO_ROOT / "data" / "raw" / "aisd" / "images"
CPAISD_SENTINEL = REPO_ROOT / "data" / "processed" / "cpaisd" / "index.csv"


def _run(script_path: Path) -> int:
    if not script_path.exists():
        print(f"  ⚠️ 스크립트 없음: {script_path}")
        return 1
    print(f"  $ python {script_path.relative_to(REPO_ROOT)}")
    return subprocess.call([sys.executable, str(script_path)])


def _has_aisd_synth() -> bool:
    p = AISD_SYNTH_SENTINEL
    return p.exists() and any(p.glob("*.png"))


def ensure_training_data(
    *,
    need_ct_hemorrhage: bool = False,  # 기본 OFF — PhysioNet 인증이라 옵션
    need_bhsd: bool = True,
    need_aisd_synth: bool = False,     # 기본 OFF — CPAISD 가 진짜 데이터
    need_cpaisd: bool = True,
) -> dict:
    """학습 진입 전에 호출. 누락된 것만 골라서 자동 다운로드/전처리.
    어떤 단계가 실패해도 다음 단계는 시도. 결과 dict 반환 (디버깅용).

    필수 (자동 준비 보장):
      - tekno21 (HuggingFace)  ← combined_dataset 가 직접 처리
      - BHSD    (HuggingFace)
      - CPAISD  (Zenodo)

    옵션 (인증/특수 사정):
      - CT Hemorrhage : PhysioNet 인증 필요. 기본 시도 안 함, 명시적 요청시만.
      - 합성 AISD     : CPAISD 가 진짜 데이터를 제공하므로 기본 비활성.
    """

    print("\n[데이터 자동 점검]")
    status: dict = {}

    # 1. CT Hemorrhage — 옵션. 명시적 요청시만 시도
    if need_ct_hemorrhage and not CT_HEM_SENTINEL.exists():
        print("  CT Hemorrhage 누락 → 자동 다운로드 시도 (옵션)")
        _run(SCRIPTS / "download_data.py")
        status["ct_hemorrhage"] = CT_HEM_SENTINEL.exists()
        if not status["ct_hemorrhage"]:
            print("  ℹ️ CT Hemorrhage 미준비 (옵션) — 학습은 다른 데이터로 계속 진행됩니다.")
            print("     사용하려면: export PHYSIONET_USER=... PHYSIONET_PASS=... 또는")
            print("                  Kaggle CLI 설정 후 재실행.")
    else:
        status["ct_hemorrhage"] = CT_HEM_SENTINEL.exists()

    # 2. BHSD — download_data.py 안에서 함께 처리되지만 직접 호출도 가능
    if need_bhsd and not BHSD_PROCESSED_SENTINEL.exists():
        print("  BHSD 전처리본 누락 → download_bhsd.py + preprocess_bhsd.py 자동 실행")
        if _run(SCRIPTS / "download_bhsd.py") == 0:
            _run(SCRIPTS / "preprocess_bhsd.py")
        status["bhsd"] = BHSD_PROCESSED_SENTINEL.exists()
    else:
        status["bhsd"] = BHSD_PROCESSED_SENTINEL.exists()

    # 3. 합성 AISD — 즉시 생성
    if need_aisd_synth and not _has_aisd_synth():
        print("  합성 AISD 누락 → generate_synthetic_aisd.py 자동 실행")
        _run(SCRIPTS / "generate_synthetic_aisd.py")
        status["aisd_synth"] = _has_aisd_synth()
    else:
        status["aisd_synth"] = _has_aisd_synth()

    # 4. CPAISD — Zenodo 자동 (preprocess 가 download 도 호출)
    if need_cpaisd and not CPAISD_SENTINEL.exists():
        print("  CPAISD 전처리본 누락 → preprocess_cpaisd.py 자동 실행 "
              "(원본 없으면 download_cpaisd.py 도 자동)")
        _run(SCRIPTS / "preprocess_cpaisd.py")
        status["cpaisd"] = CPAISD_SENTINEL.exists()
    else:
        status["cpaisd"] = CPAISD_SENTINEL.exists()

    print(f"\n  데이터 점검 결과: {status}\n")
    return status
