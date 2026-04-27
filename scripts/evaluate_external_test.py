"""외부 3-class 테스트셋 평가 (학습에 안 쓴 hold-out).

데이터: data/raw/external_test_3class/{normal,ischemic,hemorrhagic}/*.{jpg,png}
이 데이터는 본 프로젝트 학습 데이터(tekno21/BHSD/CPAISD)와 무관 → 진짜 일반화 평가.

원본이 없으면 download_external_test.py 자동 호출.

출력:
    results/external_3class/
        metrics.txt           3×3 confusion matrix + per-class precision/recall/F1
        summary.csv           샘플별 GT/예측/확률
        correct/{class}/      각 정답 버킷 시각화
        wrong/{gt}_to_{pred}/ 오분류 버킷별 시각화

실행:
    python scripts/evaluate_external_test.py
    python scripts/evaluate_external_test.py --max-per-bucket 30
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
from tqdm import tqdm

from inference.pipeline import StrokePipeline
from scripts._eval_common import classify_bucket, save_3panel, ensure_bucket_dirs


SRC_ROOT = Path("./data/raw/external_test_3class")
OUT_ROOT = Path("./results/external_3class")
CLS_CKPT = "./checkpoints/classifier/best_classifier.pth"
SEG_CKPT = "./checkpoints/segmentor/best_segmentor.pth"

CLASSES = ["normal", "ischemic", "hemorrhagic"]


def _ensure_test_set() -> bool:
    if all((SRC_ROOT / c).exists() and any((SRC_ROOT / c).glob("*"))
           for c in CLASSES):
        return True
    print("외부 테스트셋 누락 → download_external_test.py 자동 실행...")
    rc = subprocess.call([sys.executable,
                          str(Path(__file__).parent / "download_external_test.py")])
    if rc != 0:
        print("\n❌ 외부 테스트셋 자동 준비 실패. 위 안내 따라 수동 다운로드.")
        return False
    return all((SRC_ROOT / c).exists() and any((SRC_ROOT / c).glob("*"))
               for c in CLASSES)


def _gather() -> list[tuple[Path, str]]:
    samples = []
    for cls in CLASSES:
        d = SRC_ROOT / cls
        if not d.exists():
            continue
        for f in sorted(d.iterdir()):
            if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                samples.append((f, cls))
    return samples


def _confusion(gts: list[str], preds: list[str]) -> np.ndarray:
    idx = {c: i for i, c in enumerate(CLASSES)}
    m = np.zeros((3, 3), dtype=np.int64)
    for g, p in zip(gts, preds):
        m[idx[g], idx[p]] += 1
    return m


def _per_class_metrics(cm: np.ndarray) -> dict:
    out = {}
    for i, name in enumerate(CLASSES):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out[name] = {"precision": precision, "recall": recall, "f1": f1,
                     "support": int(cm[i, :].sum())}
    return out


def _write_metrics(out: Path, cm: np.ndarray, accuracy: float, per_class: dict, n_total: int):
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("외부 3-class 테스트셋 평가 결과 (학습에 사용되지 않은 hold-out)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"전체 샘플 수 : {n_total}\n")
        f.write(f"전체 정확도  : {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")

        f.write("Confusion matrix (행=GT, 열=Pred):\n")
        f.write("              " + "  ".join(f"{c:>13}" for c in CLASSES) + "\n")
        for i, c in enumerate(CLASSES):
            f.write(f"  {c:>11}  " + "  ".join(f"{v:>13d}" for v in cm[i]) + "\n")

        f.write("\nPer-class metrics:\n")
        f.write(f"  {'class':<13} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10}\n")
        for c in CLASSES:
            m = per_class[c]
            f.write(f"  {c:<13} {m['precision']:>10.4f} {m['recall']:>10.4f} "
                    f"{m['f1']:>10.4f} {m['support']:>10d}\n")


def main(args):
    if not _ensure_test_set():
        return 1

    if not Path(CLS_CKPT).exists():
        print(f"❌ 분류기 체크포인트 없음: {CLS_CKPT}")
        print(f"   먼저 학습: python training/train_classifier.py")
        return 1

    print(f"파이프라인 로드 (classifier={CLS_CKPT}"
          f"{', segmentor=' + SEG_CKPT if Path(SEG_CKPT).exists() else ''})")
    pipe = StrokePipeline(
        classifier_ckpt=CLS_CKPT,
        segmentor_ckpt=SEG_CKPT if Path(SEG_CKPT).exists() else None,
    )

    samples = _gather()
    if not samples:
        print(f"❌ 외부 테스트 이미지 0장. 확인: ls {SRC_ROOT}")
        return 1
    print(f"외부 테스트 샘플: {len(samples)} (클래스별 분포 — "
          f"{', '.join(f'{c}={sum(1 for _, g in samples if g==c)}' for c in CLASSES)})")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_bucket_dirs(OUT_ROOT, include_ischemic_correct=True)
    bucket_count = {}

    rows = []
    gts, preds = [], []

    for img_path, gt in tqdm(samples, desc="평가"):
        try:
            result = pipe.run(img_path)
        except Exception as e:
            print(f"\n  ⚠️ {img_path}: {e}")
            continue
        pred = result.class_name
        gts.append(gt)
        preds.append(pred)
        rows.append([str(img_path), gt, pred, f"{result.confidence:.4f}",
                     *(f"{result.class_probs.get(c, 0.0):.4f}" for c in CLASSES)])

        bucket = classify_bucket(gt, pred, has_ischemic_gt=True)
        bucket_count[bucket] = bucket_count.get(bucket, 0) + 1
        if bucket_count[bucket] <= args.max_per_bucket:
            orig_np = np.array(Image.open(img_path).convert("RGB"))
            out_png = OUT_ROOT / bucket / f"{img_path.stem}.png"
            try:
                save_3panel(orig_np, result, out_png, gt, dpi=80,
                            suptitle_prefix=f"[{img_path.parent.name}]")
            except Exception:
                pass

    cm = _confusion(gts, preds)
    accuracy = (cm.diagonal().sum() / cm.sum()) if cm.sum() else 0.0
    per_class = _per_class_metrics(cm)

    _write_metrics(OUT_ROOT / "metrics.txt", cm, accuracy, per_class, len(rows))

    with open(OUT_ROOT / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "gt", "pred", "confidence",
                    "p_normal", "p_ischemic", "p_hemorrhagic"])
        w.writerows(rows)

    print(f"\n결과: {OUT_ROOT}")
    print(f"  metrics.txt  — 3×3 confusion + per-class")
    print(f"  summary.csv  — {len(rows)} 행")
    print(f"  buckets      — {sum(bucket_count.values())} 시각화 (각 버킷 최대 {args.max_per_bucket}장)")
    print(f"\n전체 정확도: {accuracy*100:.2f}%")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-bucket", type=int, default=20,
                        help="버킷별 시각화 최대 장수 (기본 20)")
    args = parser.parse_args()
    sys.exit(main(args))
