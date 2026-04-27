"""results/test_full/ 를 데이터셋이 최상위인 구조로 재정리.

Before:
  correct/normal/{dataset}/...
  wrong/{dataset}/...
  hemorrhagic_to_ischemic/{dataset}/...
  ischemic_to_hemorrhagic/{dataset}/...

After (사용자 요청):
  {dataset}/correct/{class}/...png
  {dataset}/wrong/{gt}_to_{pred}/...png   (6 confusion buckets)
  {dataset}/metrics.txt   (데이터셋별 confusion + 지표)

상위 metrics.txt / summary.csv 는 그대로 유지.
"""

import csv
import shutil
import sys
from pathlib import Path

import numpy as np

OUT_ROOT = Path("./results/test_full")
SUMMARY_CSV = OUT_ROOT / "summary.csv"

CLASSES = ["normal", "ischemic", "hemorrhagic"]
DATASETS = ["brain_test", "stroke_test_3class", "external_test_3class"]
# 모든 (gt, pred) 오분류 조합 — confusion matrix 의 off-diagonal 6칸
WRONG_BUCKETS = [
    ("hemorrhagic", "ischemic"),
    ("ischemic", "hemorrhagic"),
    ("normal", "hemorrhagic"),
    ("normal", "ischemic"),
    ("ischemic", "normal"),
    ("hemorrhagic", "normal"),
]


def normalize_source(src: str) -> str:
    if src == "brain_test":
        return "brain_test"
    if src.startswith("Test/"):
        return "stroke_test_3class"
    if src.startswith("external_test_3class") or src.startswith("Brain_Stroke_CT_Dataset"):
        return "external_test_3class"
    return src.replace("/", "_")


def make_stem(img_path: str, gt: str, pred: str) -> str:
    base = Path(img_path).stem
    stem = f"{base}_{gt}_to_{pred}"
    return stem.replace(" ", "_").replace("(", "").replace(")", "")


def reorganize() -> dict:
    """기존 카테고리-우선 구조 → 데이터셋-우선 구조로 이동.
    반환: 데이터셋별 결과 row 모음 (metrics 생성용)."""
    rows_by_ds: dict = {ds: [] for ds in DATASETS}

    with open(SUMMARY_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt = row["gt"]
            pred = row["pred"]
            ds = normalize_source(row["source"])
            stem = make_stem(row["path"], gt, pred)
            fname = f"{stem}.png"

            rows_by_ds.setdefault(ds, []).append(row)

            # correct → {ds}/correct/{gt}/
            # wrong   → {ds}/wrong/{gt}_to_{pred}/   (6 confusion buckets)
            # test_full.py 직후엔 flat (correct/<gt>/, wrong/, hemorrhagic_to_ischemic/, ischemic_to_hemorrhagic/)
            if gt == pred:
                old_candidates = [
                    OUT_ROOT / "correct" / gt / fname,
                    OUT_ROOT / "correct" / gt / ds / fname,
                ]
                new_dir = OUT_ROOT / ds / "correct" / gt
            else:
                old_candidates = [
                    OUT_ROOT / "wrong" / fname,
                    OUT_ROOT / "wrong" / ds / fname,
                    # test_full.py 의 강조 복사본도 입력으로 흡수 (출혈↔허혈만)
                    OUT_ROOT / f"{gt}_to_{pred}" / fname,
                    OUT_ROOT / f"{gt}_to_{pred}" / ds / fname,
                ]
                new_dir = OUT_ROOT / ds / "wrong" / f"{gt}_to_{pred}"

            new_dir.mkdir(parents=True, exist_ok=True)
            new_path = new_dir / fname

            old = next((p for p in old_candidates if p.exists()), None)
            if old is not None and not new_path.exists():
                shutil.move(str(old), str(new_path))

    # 옛 카테고리 폴더 정리.
    #   correct/, wrong/ : 파일은 모두 옮겨졌으므로 빈 디렉토리만 rmdir.
    #   강조 폴더 (hem↔isc 등) : test_full.py 가 wrong/ 의 복사본을 따로 떨궈둠.
    #     이미 {ds}/wrong/{gt}_to_{pred}/ 에 원본이 있으므로 통째로 rmtree.
    for old_root in ("correct", "wrong"):
        p = OUT_ROOT / old_root
        if p.exists():
            for sub in sorted(p.rglob("*"), key=lambda x: -len(str(x))):
                if sub.is_dir():
                    try:
                        sub.rmdir()
                    except OSError:
                        pass
            try:
                p.rmdir()
            except OSError:
                pass

    for emp_root in ("hemorrhagic_to_ischemic", "ischemic_to_hemorrhagic",
                      "normal_to_hemorrhagic", "normal_to_ischemic",
                      "ischemic_to_normal", "hemorrhagic_to_normal"):
        p = OUT_ROOT / emp_root
        if p.exists():
            shutil.rmtree(p)

    return rows_by_ds


def write_dataset_metrics(rows_by_ds: dict):
    """데이터셋별 metrics.txt 생성 — 사용자가 폴더 들어가면 바로 보이게."""
    for ds, rows in rows_by_ds.items():
        if not rows:
            continue
        out = OUT_ROOT / ds / "metrics.txt"
        out.parent.mkdir(parents=True, exist_ok=True)

        idx = {c: i for i, c in enumerate(CLASSES)}
        cm = np.zeros((3, 3), dtype=np.int64)
        for r in rows:
            cm[idx[r["gt"]], idx[r["pred"]]] += 1

        n = int(cm.sum())
        acc = float(cm.diagonal().sum() / n) if n else 0.0

        with open(out, "w") as f:
            f.write(f"데이터셋: {ds}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"전체 샘플 : {n}\n")
            f.write(f"정확도    : {acc:.4f} ({acc*100:.2f}%)\n")
            f.write(f"오류율    : {1-acc:.4f} ({(1-acc)*100:.2f}%)\n\n")

            f.write("[Confusion matrix] (행=GT, 열=Pred)\n")
            f.write(f"              {' '.join(f'{c:>13}' for c in CLASSES)}\n")
            for i, c in enumerate(CLASSES):
                f.write(f"  {c:>11}  {' '.join(f'{v:>13d}' for v in cm[i])}\n")
            f.write("\n")

            f.write("[Per-class 지표]\n")
            f.write(f"  {'class':<13} {'precision':>10} {'recall':>10} {'F1':>10} {'support':>8}\n")
            for i, c in enumerate(CLASSES):
                tp = int(cm[i, i])
                fn = int(cm[i, :].sum() - tp)
                fp = int(cm[:, i].sum() - tp)
                support = int(cm[i, :].sum())
                if support == 0:
                    f.write(f"  {c:<13} {'(없음)':>10} {'(없음)':>10} {'(없음)':>10} {0:>8d}\n")
                    continue
                p = tp / (tp + fp) if (tp + fp) else 0.0
                r = tp / (tp + fn) if (tp + fn) else 0.0
                f1 = 2 * p * r / (p + r) if (p + r) else 0.0
                f.write(f"  {c:<13} {p:>10.4f} {r:>10.4f} {f1:>10.4f} {support:>8d}\n")
            f.write("\n")

            # 폴더 카운트
            f.write("[이 데이터셋 폴더 안 결과 개수]\n")
            ds_dir = OUT_ROOT / ds
            subs = ["correct/normal", "correct/ischemic", "correct/hemorrhagic"]
            subs += [f"wrong/{g}_to_{p}" for g, p in WRONG_BUCKETS]
            for sub in subs:
                d = ds_dir / sub
                cnt = len(list(d.glob("*.png"))) if d.exists() else 0
                f.write(f"  {sub:<35} {cnt:>6d}\n")


def main():
    if not SUMMARY_CSV.exists():
        print(f"❌ summary.csv 없음: {SUMMARY_CSV}")
        return 1

    rows_by_ds = reorganize()
    write_dataset_metrics(rows_by_ds)

    print("\n재정리 완료. 새 구조:\n")
    for ds in DATASETS:
        d = OUT_ROOT / ds
        if not d.exists():
            continue
        print(f"  {ds}/")
        subs = ["correct/normal", "correct/ischemic", "correct/hemorrhagic"]
        subs += [f"wrong/{g}_to_{p}" for g, p in WRONG_BUCKETS]
        for sub in subs:
            p = d / sub
            cnt = len(list(p.glob("*.png"))) if p.exists() else 0
            mark = " " if cnt > 0 else "·"
            print(f"    {mark} {sub:<35} {cnt:>6d}")
        print(f"      metrics.txt")
        print()

    print(f"전체 요약: {OUT_ROOT}/metrics.txt + summary.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
