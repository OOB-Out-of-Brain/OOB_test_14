"""외부 3-class 테스트셋 다운로드 (학습에 안 쓴 진짜 hold-out 데이터).

대상: Kaggle `ozguraslank/brain-stroke-ct-dataset`
   - normal / ischemic / hemorrhagic 3 클래스
   - 약 6,653 장 (normal 4,428 / ischemic 1,131 / hemorrhagic 1,094)
   - 본 프로젝트 학습에 사용한 데이터(tekno21/BHSD/CPAISD/CT Hem)와 무관 → 누수 없음

요구:
    pip install kaggle
    ~/.kaggle/kaggle.json (Kaggle Account → API → Create New Token)

출력:
    data/raw/external_test_3class/
        normal/        *.jpg|png
        ischemic/      *.jpg|png
        hemorrhagic/   *.jpg|png

스크립트는 다운로드 후 클래스 폴더 자동 표준화 (대소문자, 단/복수, 영/터키어 변형 매핑).
"""

import sys
import shutil
from pathlib import Path

OUT_DIR = Path("./data/raw/external_test_3class")
KAGGLE_REF = "ozguraslank/brain-stroke-ct-dataset"

# 다양한 폴더명 → 표준 클래스명 매핑
CLASS_ALIASES = {
    "normal": "normal", "no_stroke": "normal", "nostroke": "normal",
    "inme_yok": "normal", "inmeyok": "normal", "non-stroke": "normal",
    "ischemic": "ischemic", "ischaemic": "ischemic", "iskemi": "ischemic",
    "ischemia": "ischemic", "ischaemia": "ischemic", "isch": "ischemic",
    "hemorrhagic": "hemorrhagic", "haemorrhagic": "hemorrhagic",
    "hemorrhage": "hemorrhagic", "haemorrhage": "hemorrhagic",
    "kanama": "hemorrhagic", "bleed": "hemorrhagic", "bleeding": "hemorrhagic",
}


def _kaggle_download() -> bool:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("❌ kaggle 패키지 없음. 설치:")
        print("     pip install kaggle")
        print("   그리고 ~/.kaggle/kaggle.json 배치 (Kaggle → Account → API)")
        return False
    try:
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        print(f"❌ Kaggle 인증 실패: {e}")
        print("   ~/.kaggle/kaggle.json 가 있는지, 권한이 600 인지 확인하세요.")
        return False

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Kaggle 다운로드: {KAGGLE_REF}")
    try:
        api.dataset_download_files(KAGGLE_REF, path=str(OUT_DIR), unzip=True, quiet=False)
        return True
    except Exception as e:
        print(f"❌ 다운로드 실패: {e}")
        return False


def _normalize_layout() -> dict:
    """다운로드된 폴더 구조를 normal/ischemic/hemorrhagic 표준 폴더로 정리.
    중첩 디렉토리/대소문자/공백 모두 흡수."""
    counts = {"normal": 0, "ischemic": 0, "hemorrhagic": 0}

    for std_name in counts:
        (OUT_DIR / std_name).mkdir(parents=True, exist_ok=True)

    # 모든 이미지 파일을 찾아 부모 폴더명으로 클래스 결정
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for img in OUT_DIR.rglob("*"):
        if not img.is_file() or img.suffix.lower() not in img_exts:
            continue
        # 표준 폴더 안에 이미 있는 것은 스킵
        if img.parent.name in counts and img.parent.parent == OUT_DIR:
            counts[img.parent.name] += 1
            continue
        # 상위 경로를 거슬러 올라가며 클래스 폴더 매칭
        target_class = None
        for ancestor in img.parents:
            key = ancestor.name.lower().replace(" ", "_").replace("-", "_")
            if key in CLASS_ALIASES:
                target_class = CLASS_ALIASES[key]
                break
        if target_class is None:
            continue
        dest = OUT_DIR / target_class / img.name
        # 중복 이름이면 부모 폴더명을 prefix 로
        if dest.exists():
            dest = OUT_DIR / target_class / f"{img.parent.name}_{img.name}"
        try:
            shutil.move(str(img), str(dest))
            counts[target_class] += 1
        except Exception:
            pass

    # 빈 중첩 디렉토리 정리
    for sub in sorted(OUT_DIR.rglob("*"), key=lambda p: -len(str(p))):
        if sub.is_dir() and sub.name not in counts and sub != OUT_DIR:
            try:
                sub.rmdir()
            except OSError:
                pass

    return counts


def main() -> int:
    print("=" * 60)
    print("  외부 3-class 테스트셋 다운로드")
    print(f"  Kaggle: {KAGGLE_REF}")
    print("=" * 60)

    # 이미 정리된 상태면 스킵
    if all((OUT_DIR / c).exists() and any((OUT_DIR / c).glob("*"))
           for c in ("normal", "ischemic", "hemorrhagic")):
        print("\n✅ 이미 준비됨. 클래스별 파일 수:")
        for c in ("normal", "ischemic", "hemorrhagic"):
            n = sum(1 for _ in (OUT_DIR / c).iterdir())
            print(f"     {c:<13}: {n}")
        return 0

    if not _kaggle_download():
        print("\n수동 다운로드 안내:")
        print(f"  1) https://www.kaggle.com/datasets/{KAGGLE_REF} 접속 후 다운로드")
        print(f"  2) zip 안의 클래스 폴더를 {OUT_DIR}/ 에 풀기")
        print(f"  3) python scripts/download_external_test.py 재실행 (정리만 수행)")
        return 1

    print("\n폴더 표준화 중 (normal/ischemic/hemorrhagic)...")
    counts = _normalize_layout()
    print(f"\n결과 — {OUT_DIR}")
    for c, n in counts.items():
        print(f"  {c:<13}: {n}")

    if all(v > 0 for v in counts.values()):
        print(f"\n✅ 외부 테스트셋 준비 완료. 평가:")
        print(f"   python scripts/evaluate_external_test.py")
        return 0
    else:
        print(f"\n⚠️ 일부 클래스가 비어있습니다. 폴더명이 예상과 다를 수 있어요.")
        print(f"   확인: ls {OUT_DIR}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
