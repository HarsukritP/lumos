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
    CAPTURE_WIDTH,
    CAPTURE_HEIGHT,
    FRAME_PATH,
    PAGE_BRIGHTNESS_MAX,
    PAGE_BRIGHTNESS_MIN,
    PAGE_BRIGHT_FRAC_MIN,
    PAGE_SCORE_MIN,
    PENDING_PATH,
    TMP_DIR,
)


class CameraError(RuntimeError):
    pass


def _kill_stale_rpicam() -> None:
    """rpicam-still occasionally wedges holding the camera sensor (seen after
    the process is interrupted mid-capture or when a long-running peer thread
    blocks the CPU). Any stale copy will make *every* subsequent capture time
    out. Cheap insurance: fire a SIGKILL at anything still named rpicam-still
    before we launch a new one."""
    try:
        subprocess.run(
            ["pkill", "-9", "-x", "rpicam-still"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
    except Exception:
        pass


def _run_rpicam(out: Path, timeout_ms: int, hard_timeout_s: float) -> subprocess.CompletedProcess:
    cmd = [
        "rpicam-still",
        "-n",
        "--immediate",
        "-o", str(out),
        "--timeout", str(timeout_ms),
        "--width", str(CAPTURE_WIDTH),
        "--height", str(CAPTURE_HEIGHT),
    ]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=hard_timeout_s,
        stdin=subprocess.DEVNULL,
    )


def capture(out_path: Path | str = FRAME_PATH, timeout_ms: int = CAMERA_TIMEOUT_MS) -> Path:
    """Block until `rpicam-still` produces a JPEG at `out_path`. Return the path.

    Auto-recovers from wedged rpicam-still processes by SIGKILLing stale
    copies and retrying once with a slightly longer warm-up."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    for attempt in (1, 2):
        t0 = time.monotonic()
        try:
            proc = _run_rpicam(out, timeout_ms=timeout_ms, hard_timeout_s=15.0)
        except subprocess.TimeoutExpired as e:
            # A stuck rpicam will keep the sensor locked until we kill it.
            _kill_stale_rpicam()
            if attempt == 2:
                raise CameraError(
                    f"rpicam-still hung for {15.0:.0f}s (killed); giving up"
                ) from e
            # Let the kernel hand the camera back, then retry.
            time.sleep(0.5)
            continue

        dt = time.monotonic() - t0
        if proc.returncode == 0 and out.exists():
            return out

        _kill_stale_rpicam()
        if attempt == 2:
            raise CameraError(
                f"rpicam-still failed ({proc.returncode}) after {dt:.1f}s: "
                f"{(proc.stderr or '')[-300:]}"
            )
        # First failure — short pause and try once more with a little more
        # warm-up time so the sensor AE/AWB has a chance to stabilize.
        time.sleep(0.3)
        timeout_ms = max(timeout_ms, 800)

    # Unreachable: both attempts either returned or raised above.
    raise CameraError("rpicam-still: exhausted retries")


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


def page_score(img: Image.Image) -> dict:
    """Compute multiple 'does this look like a book page/cover?' metrics.

    Returns dict with: laplacian_var, mean, bright_frac, dark_frac.

    Calibration (128x128 grayscale):
      - random desk / dark room:   mean~100, bright_frac<0.16, dark_frac>0.5
      - book page under lamp:      mean>130, bright_frac>0.35, dark_frac<0.25
      - book cover (may be dark):   bright_frac>0.20 typically
      - blank white wall:          mean>200, bright_frac>0.7, laplacian_var<15
    """
    gray = np.asarray(img.convert("L").resize((128, 128), Image.BILINEAR), dtype=np.float32)
    mean = float(gray.mean())
    bright_frac = float((gray > 160).sum()) / gray.size
    dark_frac = float((gray < 80).sum()) / gray.size
    c = gray[1:-1, 1:-1]
    up = gray[:-2, 1:-1]
    dn = gray[2:, 1:-1]
    lt = gray[1:-1, :-2]
    rt = gray[1:-1, 2:]
    lap = -4.0 * c + up + dn + lt + rt
    return {
        "laplacian_var": float(lap.var()),
        "mean": mean,
        "bright_frac": bright_frac,
        "dark_frac": dark_frac,
    }


def is_likely_page(img: Image.Image) -> tuple[bool, str]:
    """Multi-gate filter that rejects obviously non-book scenes.

    Returns (ok, reason). Caller can log the reason at DEBUG level.

    Gates:
      1. Mean brightness in [PAGE_BRIGHTNESS_MIN, PAGE_BRIGHTNESS_MAX]
      2. Bright-pixel fraction >= PAGE_BRIGHT_FRAC_MIN (paper detection)
      3. Laplacian variance >= PAGE_SCORE_MIN (not a blank wall)
    """
    scores = page_score(img)
    var = scores["laplacian_var"]
    mean = scores["mean"]
    bf = scores["bright_frac"]

    tag = f"var={var:.0f} mean={mean:.0f} bright={bf:.0%}"

    if mean < PAGE_BRIGHTNESS_MIN:
        return False, f"too dark ({tag})"
    if mean > PAGE_BRIGHTNESS_MAX:
        return False, f"too bright ({tag})"
    if bf < PAGE_BRIGHT_FRAC_MIN:
        return False, f"no paper ({tag})"
    if var < PAGE_SCORE_MIN:
        return False, f"too blank ({tag})"
    return True, f"ok ({tag})"


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
    sc = page_score(a)
    ok_a, why_a = is_likely_page(a)
    print(f"page_score A: var={sc['laplacian_var']:.1f} mean={sc['mean']:.1f}"
          f" bright={sc['bright_frac']:.0%} dark={sc['dark_frac']:.0%}"
          f"  likely_page={ok_a} ({why_a})")
    sys.exit(0)
