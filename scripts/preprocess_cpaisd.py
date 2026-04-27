"""CPAISD .npz 슬라이스를 2D PNG (이미지 + 허혈 binary 마스크)로 변환.

원본 구조:
    data/raw/cpaisd/dataset/{train,val,test}/Study_*/Slice_*/
        ├─ raw.dcm           (사용 안 함)
        ├─ image.npz         (NumPy 슬라이스)
        ├─ mask.npz          (core/penumbra 마스크)
        └─ metadata.json

전처리:
  1. image.npz 안의 2D 배열을 자동 탐지 (다양한 키/형상 대응)
  2. HU 값처럼 보이면 brain window (center=40, width=80) 적용 → uint8
     이미 0-255 범위면 그대로
  3. mask.npz: 클래스 인덱스(0/1/2 = bg/core/penumbra) 또는
     멀티 채널(core, penumbra) 모두 대응 → core ∪ penumbra 를 binary 1
  4. 허혈 픽셀이 있는 슬라이스만 저장 (분할 학습은 lesion 있는 쌍이 필요)

원본이 없으면 download_cpaisd.py 를 자동 호출.

출력:
    data/processed/cpaisd/
        images/{study_id}_s{slice_id}.png    (H×W uint8)
        masks/{study_id}_s{slice_id}.png      (H×W binary 0/255)
        index.csv                             (image_path, mask_path, lesion_px, split)

실행:
    python scripts/preprocess_cpaisd.py
"""

import sys
import csv
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
from tqdm import tqdm


SRC_ROOT = Path("./data/raw/cpaisd/dataset")
OUT_DIR = Path("./data/processed/cpaisd")
OUT_IMG = OUT_DIR / "images"
OUT_MASK = OUT_DIR / "masks"
INDEX_CSV = OUT_DIR / "index.csv"

# 의료 표준 brain window (BHSD 와 동일)
WINDOW_CENTER = 40
WINDOW_WIDTH = 80


def _ensure_source() -> bool:
    """원본 압축 해제본이 없으면 download_cpaisd.py 자동 실행."""
    if SRC_ROOT.exists() and any(SRC_ROOT.rglob("Study_*")):
        return True
    print(f"⚠️ 원본 없음: {SRC_ROOT}")
    print(f"   download_cpaisd.py 자동 실행...\n")
    dl = Path(__file__).parent / "download_cpaisd.py"
    rc = subprocess.call([sys.executable, str(dl)])
    if rc != 0:
        print(f"\n❌ 다운로드 실패. 위 로그 확인 후 재시도:")
        print(f"   python scripts/download_cpaisd.py")
        return False
    return SRC_ROOT.exists() and any(SRC_ROOT.rglob("Study_*"))


def _pick_2d_array(npz_path: Path) -> np.ndarray | None:
    """npz 안에서 가장 그럴듯한 2D/3D 배열 자동 선택.
    - 1개 키만 있으면 그걸 사용
    - 'image'/'mask'/'data'/'arr_0' 등 흔한 이름 우선
    - 그 외엔 가장 큰 배열 선택"""
    try:
        npz = np.load(npz_path, allow_pickle=False)
    except Exception as e:
        print(f"  ⚠️ npz 로드 실패 {npz_path}: {e}")
        return None
    keys = list(npz.files)
    if not keys:
        return None
    if len(keys) == 1:
        return npz[keys[0]]
    for name in ("image", "img", "mask", "label", "data", "arr_0"):
        if name in keys:
            return npz[name]
    # fall back: 가장 큰 배열
    return max((npz[k] for k in keys), key=lambda a: a.size)


def _to_uint8_image(arr: np.ndarray) -> np.ndarray:
    """2D 슬라이스를 uint8 (H, W) 로 변환.
    - 이미 0~255 uint8 → 그대로
    - 그 외(HU 추정) → brain window 적용
    """
    arr = np.asarray(arr).squeeze()
    if arr.ndim == 3:
        # (H, W, C) 면 첫 채널, (C, H, W) 면 첫 채널
        if arr.shape[-1] in (1, 3):
            arr = arr[..., 0]
        elif arr.shape[0] in (1, 3):
            arr = arr[0]
        else:
            arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"image 배열이 2D 가 아님: shape={arr.shape}")

    arr_f = arr.astype(np.float32)
    vmin, vmax = float(arr_f.min()), float(arr_f.max())

    # 휴리스틱: 이미 0-255 정수 범위로 정규화돼 있으면 그대로
    if vmin >= 0 and vmax <= 255 and arr.dtype in (np.uint8, np.int16, np.int32):
        return arr.astype(np.uint8)

    # HU 추정 — brain window
    lo = WINDOW_CENTER - WINDOW_WIDTH / 2.0
    hi = WINDOW_CENTER + WINDOW_WIDTH / 2.0
    clipped = np.clip(arr_f, lo, hi)
    return ((clipped - lo) / (hi - lo) * 255.0).astype(np.uint8)


def _to_binary_mask(arr: np.ndarray) -> np.ndarray:
    """core/penumbra 마스크를 단일 binary (H, W) uint8 (0/1) 로 변환.
    여러 인코딩 자동 처리:
      - (H, W) class indices (0=bg, 1=core, 2=penumbra)
      - (H, W) binary (0/1 또는 0/255)
      - (2, H, W) / (H, W, 2): core, penumbra 채널 분리
    """
    arr = np.asarray(arr).squeeze()
    if arr.ndim == 2:
        return (arr > 0).astype(np.uint8)
    if arr.ndim == 3:
        # 채널 축 추정
        if arr.shape[0] <= 4 and arr.shape[0] < arr.shape[-1]:
            channels = [arr[c] for c in range(arr.shape[0])]
        elif arr.shape[-1] <= 4 and arr.shape[-1] < arr.shape[0]:
            channels = [arr[..., c] for c in range(arr.shape[-1])]
        else:
            channels = [arr[..., 0]]
        merged = np.zeros_like(channels[0], dtype=np.uint8)
        for ch in channels:
            merged |= (ch > 0).astype(np.uint8)
        return merged
    raise ValueError(f"mask 배열 shape 예상 외: {arr.shape}")


def _iter_slice_dirs():
    """split 별로 Study_*/Slice_* 디렉토리 yield."""
    for split_dir in sorted(SRC_ROOT.iterdir()):
        if not split_dir.is_dir():
            continue
        split = split_dir.name  # train/val/test
        for study_dir in sorted(split_dir.glob("Study_*")):
            for slice_dir in sorted(study_dir.glob("Slice_*")):
                yield split, study_dir.name, slice_dir.name, slice_dir


def main() -> int:
    if not _ensure_source():
        return 1

    OUT_IMG.mkdir(parents=True, exist_ok=True)
    OUT_MASK.mkdir(parents=True, exist_ok=True)

    slice_list = list(_iter_slice_dirs())
    print(f"CPAISD 슬라이스 후보: {len(slice_list)}")

    rows = []
    skipped_no_npz = 0
    skipped_empty = 0
    skipped_error = 0

    for split, study, slc, slc_dir in tqdm(slice_list, desc="처리"):
        img_npz = slc_dir / "image.npz"
        msk_npz = slc_dir / "mask.npz"
        if not (img_npz.exists() and msk_npz.exists()):
            skipped_no_npz += 1
            continue
        try:
            img_arr = _pick_2d_array(img_npz)
            msk_arr = _pick_2d_array(msk_npz)
            if img_arr is None or msk_arr is None:
                skipped_error += 1
                continue
            img_u8 = _to_uint8_image(img_arr)
            msk_bin = _to_binary_mask(msk_arr)

            # 모양 불일치는 마스크를 이미지에 맞춰 NN-resize
            if img_u8.shape != msk_bin.shape:
                from PIL import Image as _PI
                msk_bin = np.array(_PI.fromarray(msk_bin * 255).resize(
                    (img_u8.shape[1], img_u8.shape[0]), _PI.NEAREST
                )) // 255

            lesion_px = int(msk_bin.sum())
            if lesion_px == 0:
                skipped_empty += 1
                continue

            study_id = study.replace("Study_", "")
            slice_id = slc.replace("Slice_", "")
            stem = f"cp_{study_id}_s{slice_id}"
            img_png = OUT_IMG / f"{stem}.png"
            msk_png = OUT_MASK / f"{stem}.png"
            Image.fromarray(img_u8).save(img_png)
            Image.fromarray((msk_bin * 255).astype(np.uint8)).save(msk_png)
            rows.append((
                str(img_png.relative_to(OUT_DIR)),
                str(msk_png.relative_to(OUT_DIR)),
                lesion_px, split, study_id,
            ))
        except Exception as e:
            skipped_error += 1
            if skipped_error <= 5:
                print(f"\n  ⚠️ {slc_dir}: {e}")

    with open(INDEX_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "mask_path", "lesion_px", "split", "study_id"])
        w.writerows(rows)

    print(f"\n완료: {OUT_DIR}")
    print(f"  허혈 슬라이스 저장 : {len(rows)}")
    print(f"  npz 누락 스킵      : {skipped_no_npz}")
    print(f"  마스크 비어서 스킵 : {skipped_empty}")
    print(f"  처리 오류 스킵     : {skipped_error}")
    if not rows:
        print(f"\n❌ 한 장도 저장 못 함. 원본 구조가 예상과 다를 수 있습니다.")
        print(f"   확인: ls {SRC_ROOT}/train/Study_*/Slice_*/ | head")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
