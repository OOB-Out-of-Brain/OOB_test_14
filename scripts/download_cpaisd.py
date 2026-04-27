"""CPAISD (Core-Penumbra Acute Ischemic Stroke Dataset) 자동 다운로드.

출처: Zenodo record 10892316 (CC-BY-4.0, 인증 불필요)
논문: arXiv:2404.02518 "CPAISD: Core-Penumbra Acute Ischemic Stroke Dataset"
크기: ~5.6GB zip (압축 해제 후 ~6GB)
구성: 112 NCCT 스캔 / 10,165 슬라이스 (train 92 / val 10 / test 10)

다운로드 실패시 자동 재개 (HTTP Range), 부분 파일 보존.
이미 받았거나 압축 해제됐으면 스킵.

실행:
    python scripts/download_cpaisd.py
"""

import sys
import zipfile
from pathlib import Path
from urllib import request as urlreq
from urllib.error import URLError, HTTPError

OUT_DIR = Path("./data/raw/cpaisd")
ZIP_PATH = OUT_DIR / "dataset.zip"
EXTRACT_DIR = OUT_DIR / "dataset"
ZENODO_URL = "https://zenodo.org/records/10892316/files/dataset.zip?download=1"
EXPECTED_MIN_SIZE = 5 * 1024 * 1024 * 1024  # 5GB sanity floor


def _format_mb(n: int) -> str:
    return f"{n / (1024 * 1024):.1f}MB"


def _print_progress(downloaded: int, total: int, bar_w: int = 30):
    if total <= 0:
        sys.stdout.write(f"\r    {_format_mb(downloaded)} 받는 중...")
    else:
        pct = downloaded / total
        filled = int(bar_w * pct)
        bar = "█" * filled + "░" * (bar_w - filled)
        sys.stdout.write(
            f"\r    [{bar}] {pct*100:5.1f}%  "
            f"{_format_mb(downloaded)} / {_format_mb(total)}"
        )
    sys.stdout.flush()


def _get_remote_size(url: str) -> int:
    """HEAD 로 파일 전체 크기 조회. 실패시 -1."""
    try:
        req = urlreq.Request(url, method="HEAD")
        with urlreq.urlopen(req, timeout=30) as resp:
            cl = resp.headers.get("Content-Length")
            return int(cl) if cl else -1
    except (URLError, HTTPError, ValueError):
        return -1


def download_with_resume(url: str, dest: Path, chunk: int = 1024 * 256) -> bool:
    """Range 헤더로 이어받기. 이미 완전히 받았으면 즉시 True."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    remote_size = _get_remote_size(url)
    if remote_size < 0:
        print("    ⚠️ 원격 크기 조회 실패 — 처음부터 받습니다.")
    else:
        print(f"    원격 파일 크기: {_format_mb(remote_size)}")

    have = dest.stat().st_size if dest.exists() else 0
    if remote_size > 0 and have == remote_size:
        print(f"    ✅ 이미 완전히 받음: {dest}")
        return True
    if have > 0:
        print(f"    부분 파일 발견 ({_format_mb(have)}) → 이어받기 시도")

    headers = {"User-Agent": "Mozilla/5.0 (cpaisd-downloader)"}
    if have > 0 and remote_size > 0:
        headers["Range"] = f"bytes={have}-"

    try:
        req = urlreq.Request(url, headers=headers)
        with urlreq.urlopen(req, timeout=60) as resp:
            # Range 미지원이면 200으로 응답 (전체 본문) → 처음부터 다시
            status = getattr(resp, "status", resp.getcode())
            if status == 200 and have > 0:
                print("    서버가 Range 미지원 → 처음부터 다시 받습니다.")
                have = 0
                dest.unlink(missing_ok=True)

            mode = "ab" if have > 0 and status == 206 else "wb"
            total = remote_size
            downloaded = have if mode == "ab" else 0

            with open(dest, mode) as f:
                while True:
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    f.write(buf)
                    downloaded += len(buf)
                    _print_progress(downloaded, total)
            print()  # newline after progress bar
    except (URLError, HTTPError) as e:
        print(f"\n    ❌ 다운로드 실패: {e}")
        print(f"    부분 파일 ({_format_mb(dest.stat().st_size if dest.exists() else 0)}) "
              f"는 보존됩니다 — 다시 실행하면 이어받습니다.")
        return False

    final_size = dest.stat().st_size
    if final_size < EXPECTED_MIN_SIZE:
        print(f"    ⚠️ 받은 파일이 비정상적으로 작음 ({_format_mb(final_size)}) — 손상됐을 수 있음.")
        return False
    if remote_size > 0 and final_size != remote_size:
        print(f"    ⚠️ 크기 불일치: 받은 {_format_mb(final_size)} vs 예상 {_format_mb(remote_size)}")
        return False
    return True


def extract(zip_path: Path, out_dir: Path) -> bool:
    """이미 풀려있으면 스킵 (Study_ 폴더 1개 이상으로 판단)."""
    if EXTRACT_DIR.exists():
        existing = sum(1 for _ in EXTRACT_DIR.rglob("Study_*"))
        if existing > 0:
            print(f"    ✅ 이미 압축 해제됨 ({existing}개 Study 폴더)")
            return True

    print(f"    압축 해제 중 → {out_dir}")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.namelist()
            total = len(members)
            for i, m in enumerate(members, 1):
                zf.extract(m, out_dir)
                if i % 500 == 0 or i == total:
                    sys.stdout.write(f"\r      {i}/{total} 파일 풀림")
                    sys.stdout.flush()
        print()
    except zipfile.BadZipFile as e:
        print(f"    ❌ zip 손상: {e}")
        print(f"    {zip_path} 를 지우고 다시 실행해 보세요.")
        return False
    return True


def main() -> int:
    print("=" * 60)
    print("  CPAISD (Core-Penumbra Acute Ischemic Stroke Dataset)")
    print("  Zenodo 10892316  ·  CC-BY-4.0  ·  ~5.6GB")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 이미 풀려있으면 곧장 종료
    if EXTRACT_DIR.exists() and any(EXTRACT_DIR.rglob("Study_*")):
        n = sum(1 for _ in EXTRACT_DIR.rglob("Study_*"))
        print(f"\n✅ CPAISD 이미 준비 완료 ({n}개 Study)")
        print(f"   다음 단계: python scripts/preprocess_cpaisd.py")
        return 0

    print(f"\n[1/2] 다운로드 (Zenodo 직링크, 인증 불필요)")
    print(f"      URL: {ZENODO_URL}")
    print(f"      → {ZIP_PATH}")
    if not download_with_resume(ZENODO_URL, ZIP_PATH):
        print(f"\n❌ 다운로드 실패. 네트워크 확인 후 같은 명령 재실행 (이어받기).")
        return 1

    print(f"\n[2/2] 압축 해제")
    if not extract(ZIP_PATH, OUT_DIR):
        return 1

    n = sum(1 for _ in EXTRACT_DIR.rglob("Study_*"))
    print(f"\n✅ 완료. Study 폴더 {n}개")
    print(f"   다음 단계: python scripts/preprocess_cpaisd.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
