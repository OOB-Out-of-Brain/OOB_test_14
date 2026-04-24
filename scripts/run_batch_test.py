"""폴더 내 이미지를 3-class 배치 추론 + 결과 저장.

사용법:
    python scripts/run_batch_test_3class.py --input-dir /Users/bari/Downloads/brain_test \
                                             --output-dir results/brain_test_3class/

파일 이름에서 GT를 추측해 비교 출력 (선택):
    - "nomal"/"normal"  → normal
    - "iskemi"/"ischem" → ischemic
    - 그 외 (EDH/ICH/SAH/SDH 등) → hemorrhagic
  --no-gt-from-name 옵션을 주면 비교하지 않고 예측만 출력.
"""

import argparse, sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from inference.pipeline import StrokePipeline
from scripts._eval_common import classify_bucket, save_3panel, ensure_bucket_dirs


def infer_gt_from_name(name: str):
    low = name.lower()
    if re.search(r"nomal|normal", low):
        return "normal"
    if re.search(r"iskemi|ischem|isch", low):
        return "ischemic"
    if re.search(r"edh|ich|sah|sdh|ihv|hemorr|bleed|출혈", low):
        return "hemorrhagic"
    return None


def _collect_images(in_dir: Path, gt_from_folder: bool):
    """이미지 경로 + (선택) GT 튜플 리스트.
    gt_from_folder=True 이면 직속 부모 폴더명을 GT로 사용:
      - hemorrhagic / hemorrhage → hemorrhagic
      - ischaemic / ischemic / iskemi / isch* → ischemic
      - normal / nomal → normal
    """
    exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    pairs = []
    if gt_from_folder:
        for p in sorted(in_dir.rglob("*")):
            if p.suffix.lower() not in exts or p.name.startswith("result_"):
                continue
            folder = p.parent.name.lower()
            if re.search(r"hemorr|bleed", folder):
                gt = "hemorrhagic"
            elif re.search(r"iscch|ischa|ischem|iskemi|isch$", folder) or "ischa" in folder or "iskemi" in folder:
                gt = "ischemic"
            elif re.search(r"normal|nomal", folder):
                gt = "normal"
            else:
                gt = None
            pairs.append((p, gt))
    else:
        for p in sorted(in_dir.iterdir()):
            if p.suffix.lower() not in exts or p.name.startswith("result_"):
                continue
            pairs.append((p, None))
    return pairs


def main(args):
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 버킷 폴더 미리 생성 (허혈 GT 있는지 여부에 따라 correct/ischemic 포함)
    ensure_bucket_dirs(out_dir, include_ischemic_correct=args.gt_from_folder)

    print("모델 로딩...")
    pipe = StrokePipeline(
        classifier_ckpt=args.cls_ckpt,
        segmentor_ckpt=args.seg_ckpt if Path(args.seg_ckpt).exists() else None,
    )
    if pipe.segmentor is None:
        print(f"  (세그멘터 ckpt 없음 → 분류만 진행)")

    pairs = _collect_images(in_dir, gt_from_folder=args.gt_from_folder)
    images = [p for p, _ in pairs]
    folder_gt = {p: g for p, g in pairs}
    if not images:
        print(f"이미지 없음: {in_dir}")
        return
    print(f"이미지 {len(images)}개 추론 → {out_dir}\n")

    header = f"{'파일':<24} {'예측':<12} {'신뢰도':>7}  {'확률(N/I/H)':<24} {'lesion px':<20}"
    if not args.no_gt_from_name:
        header = f"{'GT':<12} " + header
    print(header)
    print("-" * len(header))

    summary = {"normal": 0, "ischemic": 0, "hemorrhagic": 0}
    cm = {}  # (gt, pred) → count
    for img_path in images:
        r = pipe.run(img_path)
        summary[r.class_name] = summary.get(r.class_name, 0) + 1

        probs_str = " / ".join(
            f"{r.class_probs.get(c, 0):.2f}"
            for c in ["normal", "ischemic", "hemorrhagic"]
        )
        lesion_str = ""
        if r.ischemic_area_px:
            lesion_str += f"isch={r.ischemic_area_px}px "
        if r.hemorrhagic_area_px:
            lesion_str += f"hem={r.hemorrhagic_area_px}px"

        # GT 소스: 폴더명(우선) → 파일명
        gt = folder_gt.get(img_path) if args.gt_from_folder else None
        if gt is None and not args.no_gt_from_name:
            gt = infer_gt_from_name(img_path.stem)
        prefix = f"{(gt or '-'):<12} " if (args.gt_from_folder or not args.no_gt_from_name) else ""
        if gt is not None:
            cm[(gt, r.class_name)] = cm.get((gt, r.class_name), 0) + 1

        print(f"{prefix}{img_path.name:<30} {r.class_name:<12} "
              f"{r.confidence:>6.1%}  {probs_str:<24} {lesion_str}")

        # 3-panel figure 저장 + 버킷별 폴더
        orig = np.array(Image.open(img_path).convert("RGB"))
        if gt is not None:
            bucket = classify_bucket(gt, r.class_name, has_ischemic_gt=args.gt_from_folder)
            # 파일명 중복 방지 위해 상대경로에서 특수문자 치환
            stem = img_path.stem.replace(" ", "_").replace("(", "").replace(")", "")
            out_path = out_dir / bucket / f"{stem}_result.png"
        else:
            out_path = out_dir / "unlabeled" / f"{img_path.stem}_result.png"
        save_3panel(orig, r, out_path, gt or "unknown", dpi=120)

    print("\n" + "=" * 70)
    print(f"  총 {len(images)}개 | normal {summary.get('normal', 0)} | "
          f"ischemic {summary.get('ischemic', 0)} | hemorrhagic {summary.get('hemorrhagic', 0)}")

    if cm:
        print("\n  라벨(파일명 기반) vs 예측:")
        classes = ["normal", "ischemic", "hemorrhagic"]
        header = "  GT\\Pred    " + "  ".join(f"{c:>12}" for c in classes)
        print(header)
        total_correct = 0
        total_seen = 0
        for g in classes:
            row = f"  {g:<10} " + "  ".join(
                f"{cm.get((g, p), 0):>12d}" for p in classes
            )
            print(row)
            for p in classes:
                total_seen += cm.get((g, p), 0)
            total_correct += cm.get((g, g), 0)
        if total_seen:
            print(f"\n  Accuracy (라벨 인식된 {total_seen}개): {total_correct / total_seen:.4f}")

    print(f"  저장: {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--cls-ckpt", default="./checkpoints/classifier/best_classifier.pth")
    p.add_argument("--seg-ckpt", default="./checkpoints/segmentor/best_segmentor.pth")
    p.add_argument("--no-gt-from-name", action="store_true",
                   help="파일명에서 GT 추측 비활성화")
    p.add_argument("--gt-from-folder", action="store_true",
                   help="하위 폴더명(hemorrhagic/ischaemic/normal)을 GT 로 사용 (rglob 스캔)")
    args = p.parse_args()
    main(args)
