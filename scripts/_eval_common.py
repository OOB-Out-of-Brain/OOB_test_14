"""테스트 스크립트 공통 유틸: 결과 분류(버킷팅) + 3-panel figure 저장.

버킷 체계 (사용자 요청):
  correct/normal/        GT=normal,      Pred=normal
  correct/ischemic/      GT=ischemic,    Pred=ischemic
  correct/hemorrhagic/   GT=hemorrhagic, Pred=hemorrhagic
  wrong/false_positive/  GT=normal,      Pred=lesion       (오탐)
  wrong/missed/          GT=lesion,      Pred=normal       (오류/놓침)
  wrong/lesion_confusion/ GT=lesion(A),  Pred=lesion(B)    (병변 타입 혼동)

ischemic GT 가 없는 데이터셋(brain_test, CQ500)에선 has_ischemic_gt=False 로 호출:
  이 때 Pred=ischemic 은 전부 wrong/false_positive/ 로 보냄 (실제 정답 불가능).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from inference.visualization import _build_figure


def classify_bucket(gt_name: str, pred_name: str, has_ischemic_gt: bool = True) -> str:
    """GT/Pred 조합을 버킷 경로(상대)로 매핑."""
    # ischemic GT 가 없는 데이터셋에서 Pred=ischemic 은 무조건 오탐 취급
    if not has_ischemic_gt and pred_name == "ischemic":
        return "wrong/false_positive"

    if gt_name == pred_name:
        return f"correct/{gt_name}"

    if gt_name == "normal" and pred_name != "normal":
        return "wrong/false_positive"  # 정상→병변 (오탐)

    if pred_name == "normal" and gt_name != "normal":
        return "wrong/missed"           # 병변→정상 (놓침/오류)

    # 병변 타입 혼동 (hem ↔ isch)
    return "wrong/lesion_confusion"


def save_3panel(orig_np, result, out_path: Path, gt_name: str,
                dpi: int = 100, suptitle_prefix: str = ""):
    """원본 + 3-class 확률바 + 병변 overlay 3-panel figure 저장."""
    if orig_np is None or result is None:
        return
    fig = _build_figure(orig_np, result, alpha=0.45)
    mark = "O" if gt_name == result.class_name else "X"
    prefix = f"{suptitle_prefix}  " if suptitle_prefix else ""
    fig.suptitle(
        f"{prefix}GT={gt_name}  ->  Pred={result.class_name.upper()} "
        f"({result.confidence:.1%}) [{mark}]",
        color="white", fontsize=14, fontweight="bold", y=0.99,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def ensure_bucket_dirs(root: Path, include_ischemic_correct: bool = True):
    """모든 버킷 폴더를 미리 만들어둔다 (빈 폴더도 생성)."""
    for sub in ["correct/normal",
                "correct/hemorrhagic",
                "wrong/false_positive",
                "wrong/missed",
                "wrong/lesion_confusion"]:
        (root / sub).mkdir(parents=True, exist_ok=True)
    if include_ischemic_correct:
        (root / "correct/ischemic").mkdir(parents=True, exist_ok=True)
