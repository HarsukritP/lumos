"""Camera capture + frame comparison helpers.

We shell out to `rpicam-still` because the `picamera2` pip package is
unreliable on this setup (per project briefing).
"""
from __future__ import annotations

import hashlib
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

from config import (
    CAMERA_ROTATION,
    CAMERA_TIMEOUT_MS,
    FRAME_PATH,
    PAGE_BRIGHTNESS_MAX,
    PAGE_BRIGHTNESS_MIN,
    PAGE_SCORE_MIN,
    PENDING_PATH,
    TMP_DIR,
)


class CameraError(RuntimeError):
    pass


def capture(out_path: Path | str = FRAME_PATH, timeout_ms: int = CAMERA_TIMEOUT_MS) -> Path:
    """Block until `rpicam-still` produces a JPEG at `out_path`. Return the path."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "rpicam-still",
        "-n",
        "--immediate",
        "-o", str(out),
        "--timeout", str(timeout_ms),
        "--width", "1536",
        "--height", "864",
    ]
    t0 = time.monotonic()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30,
        stdin=subprocess.DEVNULL,
    )
    dt = time.monotonic() - t0
    if proc.returncode != 0 or not out.exists():
        raise CameraError(
            f"rpicam-still failed ({proc.returncode}) after {dt:.1f}s: {proc.stderr[-300:]}"
        )
    return out


def load_oriented(path: Path | str) -> Image.Image:
    """Open JPEG, apply the configured rotation, return PIL Image."""
    img = Image.open(path).convert("RGB")
    if CAMERA_ROTATION:
        img = img.rotate(-CAMERA_ROTATION, expand=True)
    return img


def _gray_small(img: Image.Image, size: int = 64) -> np.ndarray:
    arr = np.asarray(img.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32)
    return arr


def similarity(a: Image.Image, b: Image.Image) -> float:
    """Normalized cross-correlation on 64x64 grayscale.
    Returns a float in approximately [-1, 1]; typically 0.9+ for identical frames,
    <0.75 for a real page turn.
    """
    x = _gray_small(a)
    y = _gray_small(b)
    x = x - x.mean()
    y = y - y.mean()
    denom = (np.sqrt((x * x).sum()) * np.sqrt((y * y).sum())) + 1e-8
    return float((x * y).sum() / denom)


def phash(img: Image.Image) -> str:
    """Short hash of a 32x32 grayscale downsample — stable per physical book."""
    arr = np.asarray(img.convert("L").resize((32, 32), Image.BILINEAR), dtype=np.uint8)
    return hashlib.md5(arr.tobytes()).hexdigest()[:16]


def page_score(img: Image.Image) -> tuple[float, float]:
    """Cheap 'does this look like a book cover/page?' metric.

    Returns (laplacian_variance, mean_brightness). Rough calibration:
      - blank wall / ceiling / dark desk: variance < 5, brightness <40 or noisy
      - book cover with art + title: variance ~ 30..120
      - open text page under the lamp: variance ~ 50..200+

    We use Laplacian variance as a proxy for edge density (text/lines)."""
    gray = np.asarray(img.convert("L").resize((128, 128), Image.BILINEAR), dtype=np.float32)
    mean = float(gray.mean())
    c = gray[1:-1, 1:-1]
    up = gray[:-2, 1:-1]
    dn = gray[2:, 1:-1]
    lt = gray[1:-1, :-2]
    rt = gray[1:-1, 2:]
    lap = -4.0 * c + up + dn + lt + rt
    return float(lap.var()), mean


def is_likely_page(img: Image.Image) -> tuple[bool, str]:
    """Return (ok, reason). Caller can log the reason at DEBUG level."""
    var, mean = page_score(img)
    if mean < PAGE_BRIGHTNESS_MIN:
        return False, f"too dark (mean={mean:.1f})"
    if mean > PAGE_BRIGHTNESS_MAX:
        return False, f"too bright (mean={mean:.1f})"
    if var < PAGE_SCORE_MIN:
        return False, f"low detail (var={var:.1f})"
    return True, f"ok (var={var:.1f}, mean={mean:.1f})"


# ----- smoke test ----------------------------------------------------------

if __name__ == "__main__":
    import sys

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    print("capturing A...")
    a_path = capture(TMP_DIR / "smoke_a.jpg")
    time.sleep(0.5)
    print("capturing B...")
    b_path = capture(TMP_DIR / "smoke_b.jpg")
    a = load_oriented(a_path)
    b = load_oriented(b_path)
    print("size:", a.size)
    print("phash A:", phash(a))
    print("phash B:", phash(b))
    print(f"similarity A vs B: {similarity(a, b):.4f}  (expect ~1.0 for same scene)")
    print(f"similarity A vs A: {similarity(a, a):.4f}  (expect 1.0)")
    var_a, mean_a = page_score(a)
    ok_a, why_a = is_likely_page(a)
    print(f"page_score A: var={var_a:.2f} mean={mean_a:.2f}  likely_page={ok_a} ({why_a})")
    sys.exit(0)
