"""
download_model.py — Download model binary from a remote URL
============================================================
The trained model (best_model.joblib, ~837 MB) is too large for git.
This script downloads it from a URL specified by the MODEL_URL environment
variable, with progress reporting and checksum verification (optional).

Usage
-----
    # Set MODEL_URL and run:
    MODEL_URL="https://your-storage/best_model.joblib" python download_model.py

    # Or call from code:
    from download_model import ensure_model
    ensure_model()  # no-op if model already exists

Environment variables
---------------------
MODEL_URL       : Required. HTTPS URL to best_model.joblib.
                  Supports SAS tokens (Azure Blob) and pre-signed URLs (S3).
MODEL_PATH      : Optional. Local destination (default: models/best_model.joblib)
MODEL_SHA256    : Optional. Expected SHA-256 hex digest to verify download.
"""

import os
import sys
import hashlib
import urllib.request
import urllib.error
from pathlib import Path

# ── Defaults ────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODEL_DEST = Path(os.environ.get("MODEL_PATH", BASE_DIR / "models" / "best_model.joblib"))
MODEL_URL  = os.environ.get("MODEL_URL", "")
EXPECTED_SHA256 = os.environ.get("MODEL_SHA256", "")


def _sha256(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while data := f.read(chunk):
            h.update(data)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    """Download url → dest with a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")

    print(f"[download_model] Downloading model …")
    print(f"  URL  : {url[:80]}{'…' if len(url) > 80 else ''}")
    print(f"  Dest : {dest}")

    downloaded = 0

    def _reporthook(block_num, block_size, total_size):
        nonlocal downloaded
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            mb  = downloaded / 1_048_576
            tot = total_size / 1_048_576
            bar = "#" * int(pct / 2)
            sys.stdout.write(f"\r  [{bar:<50}] {pct:5.1f}%  {mb:.1f}/{tot:.1f} MB")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, tmp, reporthook=_reporthook)
        print()  # newline after progress bar
    except urllib.error.URLError as e:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"Download failed: {e}") from e

    tmp.rename(dest)
    size_mb = dest.stat().st_size / 1_048_576
    print(f"[download_model] Saved {size_mb:.1f} MB → {dest}")


def ensure_model(url: str = MODEL_URL, dest: Path = MODEL_DEST) -> Path:
    """
    Ensure the model file exists at `dest`.
    If missing, download from `url`.
    Raises RuntimeError if model is still missing after attempting download.
    """
    if dest.exists():
        print(f"[download_model] Model already present: {dest} "
              f"({dest.stat().st_size / 1_048_576:.1f} MB)")
        return dest

    if not url:
        raise RuntimeError(
            f"Model not found at '{dest}' and MODEL_URL is not set.\n"
            "Options:\n"
            "  1. Mount the model file:  -v /host/models:/app/models\n"
            "  2. Set MODEL_URL env var: MODEL_URL=https://... python download_model.py\n"
            "  3. Copy manually:         cp best_model.joblib models/"
        )

    _download(url, dest)

    # Optional checksum verification
    if EXPECTED_SHA256:
        actual = _sha256(dest)
        if actual != EXPECTED_SHA256:
            dest.unlink()
            raise RuntimeError(
                f"Checksum mismatch!\n"
                f"  Expected : {EXPECTED_SHA256}\n"
                f"  Got      : {actual}\n"
                "File deleted. Re-check MODEL_URL or MODEL_SHA256."
            )
        print(f"[download_model] SHA-256 verified OK.")

    return dest


if __name__ == "__main__":
    try:
        path = ensure_model()
        print(f"[download_model] Model ready at: {path}")
        sys.exit(0)
    except RuntimeError as e:
        print(f"\n[download_model] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
