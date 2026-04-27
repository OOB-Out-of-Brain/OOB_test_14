"""통합 외부 테스트 — brain_test + stroke_test_3class 합쳐서 3-class 평가.

사용 데이터 (둘 다 학습 미사용):
  1. /Users/bari/Downloads/brain_test/         파일명에서 GT (EDH/ICH/SAH/SDH→hem, nomal→normal)
  2. data/raw/stroke_test_3class/.../Test/     폴더명에서 GT (hemorrhagic / ischaemic)

출력 (사용자 요구 폴더 구조):

    results/test_full/
        correct/                            # 맞춘 것
            normal/         정상-정상
            ischemic/       허혈-허혈
            hemorrhagic/    출혈-출혈
        wrong/                              # 못 맞춘 것 (모든 오분류 모음)
        hemorrhagic_to_ischemic/            # 출혈을 허혈이라 한 것
        ischemic_to_hemorrhagic/            # 허혈을 출혈이라 한 것
        metrics.txt                         # 정확도 + 오탐/오류율 표
        summary.csv                         # 샘플별 GT/예측/확률

실행:
    python scripts/test_full.py
"""

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
from tqdm import tqdm

from inference.pipeline import StrokePipeline
from scripts._eval_common import save_3panel


CLS_CKPT = "./checkpoints/classifier/best_classifier.pth"
SEG_CKPT = "./checkpoints/segmentor/best_segmentor.pth"

OUT_ROOT = Path("./results/test_full")

DEFAULT_SOURCES = [
    # (path, gt_from)
    (Path("/Users/bari/Downloads/brain_test"), "filename"),
    (Path("./data/raw/external_test_3class"), "folder"),  # Kaggle 3-class 의사 라벨
    # stroke_test_3class 제외: 환자 단위 라벨이라 슬라이스 단위 분류기 평가에 부적합 (2026-04-28).
    # (Path("./data/raw/stroke_test_3class/Brain_Stroke_CT-SCAN_image/Test"), "folder"),
]

CLASSES = ["normal", "ischemic", "hemorrhagic"]
EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def gt_from_filename(name: str):
    low = name.lower()
    if re.search(r"nomal|normal", low):
        return "normal"
    if re.search(r"iskemi|ischem|isch", low):
        return "ischemic"
    if re.search(r"edh|ich|sah|sdh|ihv|hemorr|bleed|kanama", low):
        return "hemorrhagic"
    return None


def gt_from_folder(folder_name: str):
    f = folder_name.lower()
    if re.search(r"hemorr|bleed|kanama", f):
        return "hemorrhagic"
    if re.search(r"ischa|ischem|iskemi", f) or f.startswith("isch"):
        return "ischemic"
    if re.search(r"normal|nomal|inme", f):
        return "normal"
    return None


def collect_samples(sources):
    """[(path, gt, source_label), ...] 반환."""
    out = []
    for src_path, gt_mode in sources:
        if not src_path.exists():
            print(f"  ⚠️ 소스 없음: {src_path}")
            continue
        if gt_mode == "filename":
            for p in sorted(src_path.iterdir()):
                if p.suffix.lower() not in EXTS or p.name.startswith("result_"):
                    continue
                gt = gt_from_filename(p.stem)
                if gt is None:
                    print(f"  (스킵, GT 추측 실패) {p.name}")
                    continue
                out.append((p, gt, src_path.name))
        elif gt_mode == "folder":
            for p in sorted(src_path.rglob("*")):
                if p.suffix.lower() not in EXTS or not p.is_file():
                    continue
                gt = gt_from_folder(p.parent.name)
                if gt is None:
                    continue
                out.append((p, gt, f"{src_path.name}/{p.parent.name}"))
    return out


def confusion_matrix(gts, preds):
    idx = {c: i for i, c in enumerate(CLASSES)}
    m = np.zeros((3, 3), dtype=np.int64)
    for g, pr in zip(gts, preds):
        m[idx[g], idx[pr]] += 1
    return m


def per_class_metrics(cm):
    out = {}
    for i, c in enumerate(CLASSES):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out[c] = {
            "tp": tp, "fn": fn, "fp": fp,
            "precision": precision, "recall": recall, "f1": f1,
            "support": int(cm[i, :].sum()),
        }
    return out


def write_metrics(out_path, cm, accuracy, per_class, n_total, source_breakdown,
                  bucket_counts):
    with open(out_path, "w") as f:
        f.write("통합 외부 테스트 평가 (학습 미사용 hold-out)\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"전체 샘플 수 : {n_total}\n")
        f.write(f"전체 정확도  : {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"오류율       : {(1-accuracy):.4f} ({(1-accuracy)*100:.2f}%)\n\n")

        f.write("[데이터 소스별 분포]\n")
        f.write(f"  {'source':<35} {'normal':>8} {'ischemic':>10} {'hemorrhagic':>13} {'합계':>8}\n")
        for src, dist in source_breakdown.items():
            f.write(f"  {src:<35} {dist['normal']:>8d} {dist['ischemic']:>10d} "
                    f"{dist['hemorrhagic']:>13d} {sum(dist.values()):>8d}\n")
        f.write("\n")

        f.write("[Confusion matrix] (행=GT, 열=Pred)\n")
        f.write(f"              {' '.join(f'{c:>13}' for c in CLASSES)}\n")
        for i, c in enumerate(CLASSES):
            f.write(f"  {c:>11}  {' '.join(f'{v:>13d}' for v in cm[i])}\n")
        f.write("\n")

        f.write("[Per-class 지표]\n")
        f.write(f"  {'class':<13} {'precision':>10} {'recall':>10} {'F1':>10} "
                f"{'TP':>6} {'FP':>6} {'FN':>6} {'support':>8}\n")
        for c in CLASSES:
            m = per_class[c]
            f.write(f"  {c:<13} {m['precision']:>10.4f} {m['recall']:>10.4f} "
                    f"{m['f1']:>10.4f} {m['tp']:>6d} {m['fp']:>6d} "
                    f"{m['fn']:>6d} {m['support']:>8d}\n")
        f.write("\n")

        f.write("[오류 유형별 (오탐/누락) 율]\n")
        # 각 오분류의 비율 (해당 GT 대비)
        f.write(f"  {'유형':<40} {'개수':>6} {'GT 대비':>10}\n")
        labels = {
            ("normal", "ischemic"):       "정상→허혈로 오탐 (False Positive ischemic)",
            ("normal", "hemorrhagic"):    "정상→출혈로 오탐 (False Positive hem)",
            ("ischemic", "normal"):       "허혈→정상으로 놓침 (False Negative ischemic)",
            ("ischemic", "hemorrhagic"):  "허혈→출혈로 혼동",
            ("hemorrhagic", "normal"):    "출혈→정상으로 놓침 (False Negative hem)",
            ("hemorrhagic", "ischemic"):  "출혈→허혈로 혼동",
        }
        for (gt, pr), label in labels.items():
            n = bucket_counts.get(f"wrong/{gt}_to_{pr}", 0)
            support = per_class[gt]["support"]
            rate = (n / support) if support else 0.0
            f.write(f"  {label:<40} {n:>6d} {rate*100:>9.2f}%\n")
        f.write("\n")

        f.write("[버킷별 시각화 폴더 카운트]\n")
        for k in sorted(bucket_counts):
            f.write(f"  {k:<40} {bucket_counts[k]:>6d}\n")


def main(args):
    if not Path(CLS_CKPT).exists():
        print(f"❌ 분류기 체크포인트 없음: {CLS_CKPT}")
        return 1

    sources = list(DEFAULT_SOURCES)
    if args.extra_dir:
        sources.append((Path(args.extra_dir), args.extra_gt_from))

    print("파이프라인 로드...")
    pipe = StrokePipeline(
        classifier_ckpt=CLS_CKPT,
        segmentor_ckpt=SEG_CKPT if Path(SEG_CKPT).exists() else None,
    )

    samples = collect_samples(sources)
    if not samples:
        print("❌ 테스트 샘플 0장")
        return 1

    # 분포 출력
    src_dist = {}
    for _, gt, src in samples:
        src_dist.setdefault(src, {"normal": 0, "ischemic": 0, "hemorrhagic": 0})
        src_dist[src][gt] += 1
    print(f"\n총 {len(samples)} 샘플")
    for src, dist in src_dist.items():
        print(f"  {src}: " + ", ".join(f"{k}={v}" for k, v in dist.items()))

    # 폴더 준비 — 사용자 요청대로 4 카테고리
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    for c in CLASSES:
        (OUT_ROOT / "correct" / c).mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "wrong").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "hemorrhagic_to_ischemic").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "ischemic_to_hemorrhagic").mkdir(parents=True, exist_ok=True)

    rows = []
    gts, preds = [], []
    bucket_counts = {}

    for img_path, gt, src in tqdm(samples, desc="평가"):
        try:
            r = pipe.run(img_path)
        except Exception as e:
            print(f"\n  ⚠️ {img_path}: {e}")
            continue
        pred = r.class_name
        gts.append(gt); preds.append(pred)

        rows.append([str(img_path), src, gt, pred, f"{r.confidence:.4f}",
                     *(f"{r.class_probs.get(c, 0.0):.4f}" for c in CLASSES)])

        # 폴더 라우팅:
        #   1) correct/<class>/  : 맞춘 경우
        #   2) wrong/            : 모든 오분류
        #   3) hem_to_isc / isc_to_hem : 강조 카테고리 (별도 복사)
        stem = f"{img_path.stem}_{gt}_to_{pred}".replace(" ", "_") \
                                                 .replace("(", "").replace(")", "")
        orig = np.array(Image.open(img_path).convert("RGB"))

        if gt == pred:
            out = OUT_ROOT / "correct" / gt / f"{stem}.png"
            bucket_counts[f"correct/{gt}"] = bucket_counts.get(f"correct/{gt}", 0) + 1
            save_3panel(orig, r, out, gt, dpi=100, suptitle_prefix=f"[{src}]")
        else:
            # wrong/ 통합
            out = OUT_ROOT / "wrong" / f"{stem}.png"
            bucket_counts[f"wrong/{gt}_to_{pred}"] = bucket_counts.get(f"wrong/{gt}_to_{pred}", 0) + 1
            save_3panel(orig, r, out, gt, dpi=100, suptitle_prefix=f"[{src}]")
            # 강조 폴더 — 출혈↔허혈 혼동
            if gt == "hemorrhagic" and pred == "ischemic":
                shutil.copyfile(out, OUT_ROOT / "hemorrhagic_to_ischemic" / out.name)
            elif gt == "ischemic" and pred == "hemorrhagic":
                shutil.copyfile(out, OUT_ROOT / "ischemic_to_hemorrhagic" / out.name)

    cm = confusion_matrix(gts, preds)
    accuracy = cm.diagonal().sum() / cm.sum() if cm.sum() else 0.0
    pcm = per_class_metrics(cm)

    write_metrics(OUT_ROOT / "metrics.txt", cm, accuracy, pcm, len(rows),
                  src_dist, bucket_counts)

    with open(OUT_ROOT / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "source", "gt", "pred", "confidence",
                    "p_normal", "p_ischemic", "p_hemorrhagic"])
        w.writerows(rows)

    # 콘솔에도 표 출력 (사용자가 바로 볼 수 있게)
    print("\n" + "=" * 70)
    print(f"전체 정확도: {accuracy*100:.2f}%   ({cm.diagonal().sum()}/{cm.sum()})")
    print(f"오류율     : {(1-accuracy)*100:.2f}%")
    print()
    print("Confusion matrix (행=GT, 열=Pred):")
    print(f"              {' '.join(f'{c:>13}' for c in CLASSES)}")
    for i, c in enumerate(CLASSES):
        print(f"  {c:>11}  {' '.join(f'{v:>13d}' for v in cm[i])}")
    print()
    print(f"  {'class':<13} {'precision':>10} {'recall':>10} {'F1':>10} {'support':>8}")
    for c in CLASSES:
        m = pcm[c]
        print(f"  {c:<13} {m['precision']:>10.4f} {m['recall']:>10.4f} "
              f"{m['f1']:>10.4f} {m['support']:>8d}")
    print()
    print("주요 혼동:")
    print(f"  출혈→허혈 (hem→isc) : {bucket_counts.get('wrong/hemorrhagic_to_ischemic', 0)}장")
    print(f"  허혈→출혈 (isc→hem) : {bucket_counts.get('wrong/ischemic_to_hemorrhagic', 0)}장")
    print()
    print(f"결과: {OUT_ROOT}")
    print(f"  metrics.txt  — 상세 표")
    print(f"  summary.csv  — 샘플별 결과")
    print(f"  correct/{{normal,ischemic,hemorrhagic}}/ — 맞춘 것")
    print(f"  wrong/  — 모든 오분류")
    print(f"  hemorrhagic_to_ischemic/  — 출혈→허혈 (강조)")
    print(f"  ischemic_to_hemorrhagic/  — 허혈→출혈 (강조)")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--extra-dir", help="추가 테스트 디렉토리 (옵션)")
    p.add_argument("--extra-gt-from", choices=["filename", "folder"], default="folder")
    args = p.parse_args()
    sys.exit(main(args))
