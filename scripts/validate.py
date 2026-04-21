"""검증 리포트 생성: 분류 + 세그멘테이션 모델 성능 평가."""

import sys, torch, numpy as np, yaml
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.ct_hemorrhage_dataset import (
    build_ct_classifier_dataloaders, build_ct_seg_dataloaders, CT_CLASS_NAMES
)
from models.classifier import StrokeClassifier
from models.segmentor import StrokeSegmentor
from training.metrics import cls_report, conf_matrix, dice_score, iou_score
from tqdm import tqdm


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ct_path = cfg["data"]["ct_hemorrhage_path"]

    print("=" * 60)
    print("  검증 리포트 (CT Hemorrhage Validation Set)")
    print("=" * 60)

    print("\n[1] 분류 모델 (EfficientNet-B2)")
    print("-" * 60)
    ckpt = torch.load("./checkpoints/classifier/best_classifier.pth",
                      map_location=device, weights_only=False)
    model = StrokeClassifier(num_classes=2, pretrained=False,
                             dropout_rate=ckpt["config"]["dropout_rate"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    _, val_loader, _ = build_ct_classifier_dataloaders(
        data_root=ct_path, image_size=224, batch_size=16
    )
    val_loader = torch.utils.data.DataLoader(
        val_loader.dataset, batch_size=16, shuffle=False, num_workers=0
    )

    preds, labels = [], []
    with torch.no_grad():
        for imgs, lbl in tqdm(val_loader, desc="  평가"):
            out = model(imgs.to(device))
            preds.append(out.argmax(dim=1).cpu().numpy())
            labels.append(lbl.numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)

    acc = (preds == labels).mean()
    print(f"\n최고 에폭      : {ckpt['epoch']}")
    print(f"검증 샘플 수   : {len(labels)}")
    print(f"정확도 (Acc)   : {acc:.4f}")
    print(f"\n클래스별 리포트:")
    print(cls_report(preds, labels, CT_CLASS_NAMES))
    cm = conf_matrix(preds, labels)
    print("혼동 행렬:")
    print(f"                  예측:normal  예측:hemorrhagic")
    print(f"실제:normal       {cm[0,0]:>10d}  {cm[0,1]:>16d}")
    print(f"실제:hemorrhagic  {cm[1,0]:>10d}  {cm[1,1]:>16d}")

    print("\n\n[2] 세그멘테이션 모델 (U-Net + ResNet34)")
    print("-" * 60)
    ckpt_s = torch.load("./checkpoints/segmentor/best_segmentor.pth",
                        map_location=device, weights_only=False)
    seg = StrokeSegmentor(encoder_name=ckpt_s["config"]["encoder"],
                          encoder_weights=None)
    seg.load_state_dict(ckpt_s["model_state"])
    seg.to(device).eval()

    _, val_loader_s = build_ct_seg_dataloaders(
        data_root=ct_path, image_size=256, batch_size=8
    )
    val_loader_s = torch.utils.data.DataLoader(
        val_loader_s.dataset, batch_size=8, shuffle=False, num_workers=0
    )

    dices, ious = [], []
    with torch.no_grad():
        for imgs, masks in tqdm(val_loader_s, desc="  평가"):
            logits = seg(imgs.to(device))
            pred = (torch.sigmoid(logits) > 0.5).float()
            dices.append(dice_score(pred, masks.to(device)))
            ious.append(iou_score(pred, masks.to(device)))

    print(f"\n최고 에폭    : {ckpt_s['epoch']}")
    print(f"검증 샘플 수 : {len(val_loader_s.dataset)}")
    print(f"Dice Score   : {np.mean(dices):.4f}")
    print(f"IoU Score    : {np.mean(ious):.4f}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
