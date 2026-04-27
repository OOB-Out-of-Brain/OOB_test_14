"""results/test_full/ 안의 결과 PNG 들을 데이터 소스별 하위 폴더로 재정리.

원래 구조:
    results/test_full/
        correct/{normal,ischemic,hemorrhagic}/*.png
        wrong/*.png
        hemorrhagic_to_ischemic/*.png
        ischemic_to_hemorrhagic/*.png

재정리 후:
    results/test_full/
        correct/
            normal/{brain_test,stroke_test_3class,external_test_3class}/*.png
            ischemic/{...}/
            hemorrhagic/{...}/
        wrong/
            {brain_test,stroke_test_3class,external_test_3class}/*.png
        hemorrhagic_to_ischemic/
            {...}/*.png
        ischemic_to_hemorrhagic/
            {...}/*.png

소스 매핑:
    "brain_test"               → brain_test
    "Test/*"                    → stroke_test_3class
    "external_test_3class/*"    → external_test_3class

재테스트 없이 summary.csv 만 읽어서 파일을 mv 함. 추론은 다시 안 돔.
"""

import csv
import re
import shutil
import sys
from pathlib import Path

OUT_ROOT = Path("./results/test_full")
SUMMARY_CSV = OUT_ROOT / "summary.csv"


def normalize_source(src: str) -> str:
    """summary.csv 의 source 값을 top-level 데이터셋 이름으로."""
    if src == "brain_test":
        return "brain_test"
    if src.startswith("Test/"):
        return "stroke_test_3class"
    if src.startswith("external_test_3class") or src.startswith("Brain_Stroke_CT_Dataset"):
        return "external_test_3class"
    # 기타 — 폴더명 그대로
    return src.replace("/", "_")


def make_stem(img_path: str, gt: str, pred: str) -> str:
    """test_full.py 가 만든 출력 파일 stem 재현."""
    base = Path(img_path).stem
    stem = f"{base}_{gt}_to_{pred}"
    return stem.replace(" ", "_").replace("(", "").replace(")", "")


def main():
    if not SUMMARY_CSV.exists():
        print(f"❌ summary.csv 없음: {SUMMARY_CSV}")
        return 1

    moved = 0
    duplicated = 0
    missing = 0
    by_bucket = {}

    with open(SUMMARY_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt = row["gt"]
            pred = row["pred"]
            src_norm = normalize_source(row["source"])
            stem = make_stem(row["path"], gt, pred)
            fname = f"{stem}.png"

            # 결정: 어떤 카테고리/소스 폴더로 옮겨야 하나
            if gt == pred:
                old = OUT_ROOT / "correct" / gt / fname
                new_dir = OUT_ROOT / "correct" / gt / src_norm
                key = f"correct/{gt}/{src_norm}"
            else:
                old = OUT_ROOT / "wrong" / fname
                new_dir = OUT_ROOT / "wrong" / src_norm
                key = f"wrong/{src_norm}"

            new_dir.mkdir(parents=True, exist_ok=True)
            new_path = new_dir / fname

            if old.exists():
                shutil.move(str(old), str(new_path))
                moved += 1
                by_bucket[key] = by_bucket.get(key, 0) + 1
            else:
                # 이미 한 번 옮겨졌거나, 출력 자체가 없거나
                if new_path.exists():
                    by_bucket[key] = by_bucket.get(key, 0) + 1
                else:
                    missing += 1

            # 강조 폴더 (출혈↔허혈 혼동)
            if gt == "hemorrhagic" and pred == "ischemic":
                emp_dir = OUT_ROOT / "hemorrhagic_to_ischemic" / src_norm
            elif gt == "ischemic" and pred == "hemorrhagic":
                emp_dir = OUT_ROOT / "ischemic_to_hemorrhagic" / src_norm
            else:
                emp_dir = None

            if emp_dir is not None:
                emp_dir.mkdir(parents=True, exist_ok=True)
                # 기존 강조 폴더에 평탄 파일이 있으면 source 폴더로 이동
                old_emp = emp_dir.parent / fname  # e.g., hemorrhagic_to_ischemic/{fname}
                if old_emp.exists():
                    shutil.move(str(old_emp), str(emp_dir / fname))
                    duplicated += 1
                # 만약 wrong 에서 옮긴 파일도 강조 폴더에 복사 (test_full.py 원본 동작 유지)
                emp_target = emp_dir / fname
                if not emp_target.exists() and new_path.exists():
                    shutil.copyfile(str(new_path), str(emp_target))
                    duplicated += 1
                key2 = f"{emp_dir.parent.name}/{src_norm}"
                by_bucket[key2] = by_bucket.get(key2, 0) + 1

    # 최상위 강조 폴더에 남아있는 평탄 파일 정리 (혹시 빠진 것)
    for emp_root in (OUT_ROOT / "hemorrhagic_to_ischemic", OUT_ROOT / "ischemic_to_hemorrhagic"):
        for f in list(emp_root.glob("*.png")):
            if f.is_file():
                # 어느 source 인지 모르므로 unknown_source 폴더로
                fallback = emp_root / "unknown_source"
                fallback.mkdir(exist_ok=True)
                shutil.move(str(f), str(fallback / f.name))

    print(f"\n재정리 완료:")
    print(f"  이동된 파일: {moved}")
    print(f"  강조 폴더 정리: {duplicated}")
    print(f"  missing(원본 PNG 없음): {missing}")
    print(f"\n버킷별 카운트 (재정리 후):")
    for k in sorted(by_bucket):
        print(f"  {k:<55} {by_bucket[k]:>6d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
