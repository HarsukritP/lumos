"""Local OCR for Lumos.

Backed by the Tesseract binary via `pytesseract`. Used for two purposes:
  1. Same-page detection without burning a Gemini call (read the printed
     page number and compare to STATE.current_page).
  2. Resume-from-last-page confirmation (has the reader flipped to the
     page the device remembers?).

We keep the OCR surface tiny and boring on purpose: downscale first, top
and bottom strips only, digits-only whitelist, single-line PSM, text-
density pre-filter so dead-blank strips short-circuit before Tesseract
even starts. Anything richer is Gemini's job.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

import pytesseract

log = logging.getLogger("lumos.ocr")


# Tesseract config:
#   --oem 1  LSTM only (Tesseract 5 default, best digit accuracy).
#   --psm 7  single text line — a header/footer strip is exactly that.
#   tessedit_char_whitelist  digits only; stops Tesseract from hallucinating
#                            letters from chapter-title bleedover.
_TESS_CONFIG = "--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789"

# Hard cap per strip. Tesseract is inherently unpredictable on frames with no
# usable text (it keeps searching); the pre-filter below avoids most of those
# calls but we still cap so one pathological crop can't freeze the watch loop.
_TESS_TIMEOUT_S = 3.5

# 1-4 digit page numbers (books rarely exceed 9999p).
_DIGIT_RE = re.compile(r"(?<!\d)(\d{1,4})(?!\d)")

# Work at ~800px wide. Tesseract's accuracy on print-size digits plateaus
# there; going larger just burns CPU on a Pi Zero 2 W.
_MAX_WIDTH_PX = 800
# Thin strips — page numbers live in the running header/footer, not deep
# into the body text.
_STRIP_FRACTION = 10

# Text-density pre-filter. After a fast Otsu threshold we compute the
# fraction of "dark" pixels in the strip. A blank paper-white strip is ~0%;
# a text-line strip is ~3-20%; a photo-heavy page is >40%. Strips outside
# this window don't get passed to Tesseract.
_DENSITY_MIN = 0.005
_DENSITY_MAX = 0.50


def _downscale(img: Image.Image, max_w: int = _MAX_WIDTH_PX) -> Image.Image:
    w, h = img.size
    if w <= max_w:
        return img
    new_h = int(h * max_w / w)
    return img.resize((max_w, new_h), Image.BICUBIC)


def _strip_density(strip_gray: Image.Image) -> float:
    """Fraction of dark pixels after Otsu-ish global threshold. Cheap."""
    arr = np.asarray(strip_gray, dtype=np.uint8)
    # Rough Otsu: split at mean - 0.5*std; close enough for header strips.
    mean = float(arr.mean())
    std = float(arr.std())
    t = max(40, min(220, mean - 0.5 * std))
    dark = (arr < t).sum()
    return dark / arr.size


def _ocr_strip(strip: Image.Image) -> str:
    gray = strip.convert("L")
    gray = ImageOps.autocontrast(gray, cutoff=2)
    density = _strip_density(gray)
    if density < _DENSITY_MIN or density > _DENSITY_MAX:
        log.debug("skip OCR (density=%.3f)", density)
        return ""
    try:
        return pytesseract.image_to_string(
            gray,
            config=_TESS_CONFIG,
            timeout=_TESS_TIMEOUT_S,
        ).strip()
    except Exception as e:
        log.debug("tesseract failed: %r", e)
        return ""


def _candidate_numbers(img: Image.Image) -> list[int]:
    """Pull candidate page numbers from the top and bottom strips."""
    small = _downscale(img)
    w, h = small.size
    strip_h = max(20, h // _STRIP_FRACTION)
    top = small.crop((0, 0, w, strip_h))
    bottom = small.crop((0, h - strip_h, w, h))
    out: list[int] = []
    for strip in (top, bottom):
        text = _ocr_strip(strip)
        if not text:
            continue
        for m in _DIGIT_RE.finditer(text):
            try:
                out.append(int(m.group(1)))
            except ValueError:
                continue
    return out


def read_page_number(img: Image.Image | Path | str) -> int | None:
    """Return a best-guess printed page number from a book-page photo, or
    None if nothing confidently readable was found.

    Strategy: OCR top + bottom strips, filter digits, pick the most-common
    candidate (tie-break with smallest value, since big numbers in running
    headers are usually chapter/year). Outliers (0 or > 9999) dropped.
    """
    if isinstance(img, (str, Path)):
        img = Image.open(img).convert("RGB")

    cands = [n for n in _candidate_numbers(img) if 1 <= n <= 9999]
    if not cands:
        return None

    counts: dict[int, int] = {}
    for n in cands:
        counts[n] = counts.get(n, 0) + 1
    best = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    log.debug("ocr candidates=%r -> %d", cands, best)
    return best


# ----- smoke test ----------------------------------------------------------

if __name__ == "__main__":
    import sys
    import time

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if len(sys.argv) < 2:
        print("Usage: python3 ocr.py <image.jpg>")
        sys.exit(1)
    path = Path(sys.argv[1])
    t0 = time.monotonic()
    pn = read_page_number(path)
    dt = time.monotonic() - t0
    print(f"page_number={pn}  ({dt*1000:.0f}ms)")
