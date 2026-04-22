"""CQ500 (qure.ai) 테스트 데이터셋 다운로드 시도.

공식 배포 방식: 이메일 등록 후 S3 pre-signed URL 발급 (자동화 불가)
대안: Academic Torrent, 공개 mirror 시도

주의: CQ500은 **테스트 전용** 입니다. 학습에 사용하지 마세요.

실행:
    python scripts/download_cq500.py
"""

import sys, urllib.request
from pathlib import Path

CQ500_DIR = Path("./data/raw/cq500")

# 시도해볼 URL 목록 (다 실패하면 수동 안내)
CANDIDATE_URLS = [
    # 공개 S3 mirror (예상, 실제로는 이메일 인증 필요할 수 있음)
    "http://headctstudy.qure.ai/storage/dataset/CQ500.zip",
    "https://s3.amazonaws.com/qureheadctstudy/CQ500.zip",
]

# reads.csv (라벨 파일)는 별도 공개
READS_CSV_URL = "https://s3.amazonaws.com/qureheadctstudy/reads.csv"


def _check(url: str, timeout: int = 10) -> bool:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return 200 <= r.status < 400
    except Exception:
        return False


def main():
    CQ500_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  CQ500 (qure.ai) 테스트 데이터셋 준비")
    print("  ⚠️  테스트 전용 — 학습에 사용하지 말 것")
    print("=" * 60)

    # 1) reads.csv (라벨) 다운로드 시도
    reads_csv = CQ500_DIR / "reads.csv"
    if not reads_csv.exists():
        print(f"\n[1] reads.csv 시도: {READS_CSV_URL}")
        if _check(READS_CSV_URL):
            try:
                urllib.request.urlretrieve(READS_CSV_URL, reads_csv)
                print(f"  ✅ 저장: {reads_csv}")
            except Exception as e:
                print(f"  ❌ 다운로드 실패: {e}")
        else:
            print(f"  ⚠️  URL 접근 불가")

    # 2) zip 후보 URL 테스트
    print(f"\n[2] CQ500 zip 후보 URL 체크...")
    for url in CANDIDATE_URLS:
        ok = _check(url)
        print(f"  {'✅' if ok else '❌'}  {url}")
        if ok:
            zip_path = CQ500_DIR / "CQ500.zip"
            print(f"\n  다운로드 시작 → {zip_path}")
            try:
                urllib.request.urlretrieve(url, zip_path)
                print(f"  ✅ 완료 (용량 매우 큼, 30GB+)")
                return
            except Exception as e:
                print(f"  ❌ 다운로드 실패: {e}")

    # 3) 전부 실패 → 수동 안내
    print(f"""
[3] 자동 다운로드 실패. 수동 받으세요:

  1. 브라우저: http://headctstudy.qure.ai/dataset
  2. 이메일 등록하면 S3 presigned URL 몇 개 이메일로 옴
  3. 링크들(Batch zip 여러 개) 다운로드 → 전부 data/raw/cq500/ 에 압축 해제
  4. 폴더 구조 예상:
       data/raw/cq500/
         CQ500CT1/Unknown Study/... *.dcm
         CQ500CT2/...
         ...
         reads.csv
  5. 이후 평가: python scripts/evaluate_cq500.py
""")


if __name__ == "__main__":
    main()
