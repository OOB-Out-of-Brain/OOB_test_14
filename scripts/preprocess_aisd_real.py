"""AISD (GriffinLiang) 진짜 NCCT + 의사 마스크 → 학습용 PNG 슬라이스.

원본 구조:
    data/raw/aisd_real/
        image/<patient_id>/000.png .. NN.png
        mask/<patient_id>/000.png .. NN.png

마스크 라벨 (5단계):
    0 = background
    1 = remote infarct (확실)
    2 = clear acute infarct (확실)
    3 = blurred acute infarct (확실)
    4 = invisible acute infarct (불확실 — 제외)
    5 = infarct (확실)

사용자 요청: "마스크가 확실한 걸로만 추가" → {1,2,3,5} 만 lesion 으로, 4 는 제외.

출력:
    data/processed/aisd_real/
        images/<pid>_s<idx>.png   (원본 이미지 그대로)
        masks/<pid>_s<idx>.png    (binary 0/255)
        index.csv                 (image_path, mask_path, lesion_px, patient_id)

lesion 있는 슬라이스만 저장.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
from tqdm import tqdm


SRC_ROOT = Path("./data/raw/aisd_real")
OUT_ROOT = Path("./data/processed/aisd_real")
OUT_IMG = OUT_ROOT / "images"
OUT_MASK = OUT_ROOT / "masks"
INDEX_CSV = OUT_ROOT / "index.csv"

# 확실한 lesion 라벨만 (사용자 요청)
LESION_VALUES = {1, 2, 3, 5}


def main() -> int:
    if not (SRC_ROOT / "image").exists() or not (SRC_ROOT / "mask").exists():
        print(f"❌ AISD 원본 없음: {SRC_ROOT}")
        print(f"   download: gdown image.zip + mask.zip from GriffinLiang/AISD")
        return 1

    OUT_IMG.mkdir(parents=True, exist_ok=True)
    OUT_MASK.mkdir(parents=True, exist_ok=True)

    patients = sorted([p.name for p in (SRC_ROOT / "image").iterdir() if p.is_dir()])
    print(f"환자: {len(patients)}")

    rows = []
    skipped_empty = 0
    skipped_only_invisible = 0
    skipped_missing = 0

    for pid in tqdm(patients, desc="처리"):
        img_dir = SRC_ROOT / "image" / pid
        msk_dir = SRC_ROOT / "mask" / pid
        if not msk_dir.exists():
            skipped_missing += 1
            continue
        for img_path in sorted(img_dir.glob("*.png")):
            msk_path = msk_dir / img_path.name
            if not msk_path.exists():
                skipped_missing += 1
                continue
            msk = np.array(Image.open(msk_path).convert("L"))

            # 확실한 lesion 픽셀만 추출
            lesion = np.isin(msk, list(LESION_VALUES))
            if not lesion.any():
                # 혹시 4 (invisible) 만 있는 슬라이스라면 그냥 비어있는 것으로 처리 (제외)
                if (msk == 4).any():
                    skipped_only_invisible += 1
                else:
                    skipped_empty += 1
                continue

            slice_idx = img_path.stem
            stem = f"{pid}_s{slice_idx}"
            out_img = OUT_IMG / f"{stem}.png"
            out_msk = OUT_MASK / f"{stem}.png"

            # 이미지는 그대로 복사 (이미 PNG, brain windowing 된 상태)
            Image.open(img_path).save(out_img)
            # 마스크는 binary 0/255 로 정규화
            Image.fromarray((lesion.astype(np.uint8) * 255)).save(out_msk)
            rows.append((
                str(out_img.relative_to(OUT_ROOT)),
                str(out_msk.relative_to(OUT_ROOT)),
                int(lesion.sum()),
                pid,
            ))

    with open(INDEX_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "mask_path", "lesion_px", "patient_id"])
        w.writerows(rows)

    print(f"\n완료: {OUT_ROOT}")
    print(f"  허혈 슬라이스 저장        : {len(rows)}")
    print(f"  비어있는 슬라이스 스킵    : {skipped_empty}")
    print(f"  invisible 만 있는 슬라이스: {skipped_only_invisible}")
    print(f"  마스크 누락 슬라이스      : {skipped_missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
