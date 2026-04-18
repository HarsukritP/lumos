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
    sys.exit(0)
