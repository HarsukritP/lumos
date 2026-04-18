"""OLED driver for Lumos (SSD1306 128x64 over I2C).

All rendering goes through a single lock so the watch loop, idle loop, and
question thread never collide on the I2C bus.
"""
from __future__ import annotations

import io
import threading
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from config import OLED_HEIGHT, OLED_I2C_ADDR, OLED_WIDTH


_FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
_FONT_REGULAR_PATH = _FONT_DIR / "DejaVuSans.ttf"
_FONT_BOLD_PATH = _FONT_DIR / "DejaVuSans-Bold.ttf"
_FONT_MONO_PATH = _FONT_DIR / "DejaVuSansMono-Bold.ttf"


def _load_font(path: Path, size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(str(path), size)
    except Exception:
        return ImageFont.load_default()


FONT_SM = _load_font(_FONT_REGULAR_PATH, 10)
FONT_MD = _load_font(_FONT_BOLD_PATH, 12)
FONT_LG = _load_font(_FONT_BOLD_PATH, 16)
FONT_XL = _load_font(_FONT_BOLD_PATH, 34)
FONT_MONO = _load_font(_FONT_MONO_PATH, 10)


_lock = threading.Lock()
_oled = None
_initialized = False


def _init_oled() -> None:
    """Bring up the SSD1306 lazily so importing display.py never crashes
    a host that has no I2C bus (e.g. dev machine)."""
    global _oled, _initialized
    if _initialized:
        return
    _initialized = True
    try:
        import board  # adafruit-blinka
        import busio
        import adafruit_ssd1306

        i2c = busio.I2C(board.SCL, board.SDA)
        _oled = adafruit_ssd1306.SSD1306_I2C(
            OLED_WIDTH, OLED_HEIGHT, i2c, addr=OLED_I2C_ADDR
        )
        _oled.fill(0)
        _oled.show()
    except Exception as e:
        print(f"[display] OLED init failed: {e!r}. Running headless.")
        _oled = None


def is_real() -> bool:
    _init_oled()
    return _oled is not None


def _blank() -> Image.Image:
    return Image.new("1", (OLED_WIDTH, OLED_HEIGHT), 0)


def _flush(img: Image.Image) -> None:
    _init_oled()
    if _oled is None:
        return
    _oled.image(img)
    _oled.show()


def _text_width(draw: ImageDraw.ImageDraw, s: str, font) -> int:
    bbox = draw.textbbox((0, 0), s, font=font)
    return bbox[2] - bbox[0]


def wrap(text: str, font, max_width_px: int) -> list[str]:
    """Greedy word-wrap that respects pixel width for the chosen font."""
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    dummy = ImageDraw.Draw(_blank())
    for w in words[1:]:
        candidate = current + " " + w
        if _text_width(dummy, candidate, font) <= max_width_px:
            current = candidate
        else:
            lines.append(current)
            current = w
    lines.append(current)
    return lines


def clear() -> None:
    with _lock:
        _init_oled()
        if _oled is None:
            return
        _oled.fill(0)
        _oled.show()


def show_text(lines: list[str], font=FONT_MD, header: str | None = None) -> None:
    """Render up to ~5 lines of text. First optional header uses the bold font."""
    img = _blank()
    draw = ImageDraw.Draw(img)
    y = 2
    if header:
        draw.text((2, y), header, font=FONT_LG, fill=255)
        y += FONT_LG.size + 2
        draw.line([(2, y), (OLED_WIDTH - 3, y)], fill=255, width=1)
        y += 3
    line_height = font.size + 1
    for line in lines:
        if y + line_height > OLED_HEIGHT:
            break
        draw.text((2, y), line, font=font, fill=255)
        y += line_height
    with _lock:
        _flush(img)


def show_status(header: str, body: str | list[str]) -> None:
    """Standard status screen: bold header, wrapped body underneath."""
    body_lines = body if isinstance(body, list) else wrap(body, FONT_MD, OLED_WIDTH - 4)
    img = _blank()
    draw = ImageDraw.Draw(img)
    draw.text((2, 2), header, font=FONT_LG, fill=255)
    draw.line([(2, 20), (OLED_WIDTH - 3, 20)], fill=255, width=1)
    y = 24
    for line in body_lines[:4]:
        draw.text((2, y), line, font=FONT_MD, fill=255)
        y += FONT_MD.size + 1
    with _lock:
        _flush(img)


def show_ready(book_title: str | None, current_page: int | None) -> None:
    if book_title and current_page:
        show_status(
            "Lumos",
            [
                "ready",
                f"picking up at",
                f"p. {current_page}",
            ],
        )
    else:
        show_status("Lumos", ["ready", "open a book"])


def show_caught_up(page_number: int | None, book_title: str | None = None) -> None:
    """Primary reading-phase idle display: book title (tiny), giant page
    number, 'caught up' footer. This is the main trust signal — if the number
    shown here matches the page the user is actually on, Lumos is in sync."""
    img = _blank()
    draw = ImageDraw.Draw(img)

    if book_title and book_title != "Unknown":
        title = book_title
        # Hand-truncate so the top row is always ~1 line.
        max_chars = 20
        if len(title) > max_chars:
            title = title[: max_chars - 1] + "\u2026"
        draw.text((2, 0), title, font=FONT_SM, fill=255)

    page_str = f"p. {page_number}" if page_number else "p. ?"
    tw = _text_width(draw, page_str, FONT_XL)
    x = max(2, (OLED_WIDTH - tw) // 2)
    # Center vertically in the middle band.
    draw.text((x, 12), page_str, font=FONT_XL, fill=255)

    # Footer
    footer = "caught up"
    fw = _text_width(draw, footer, FONT_SM)
    draw.text(((OLED_WIDTH - fw) // 2, OLED_HEIGHT - FONT_SM.size - 2),
              footer, font=FONT_SM, fill=255)

    with _lock:
        _flush(img)


def show_page_summary(page_number: int, summary: str) -> None:
    body = wrap(summary, FONT_SM, OLED_WIDTH - 4)
    img = _blank()
    draw = ImageDraw.Draw(img)
    draw.text((2, 2), f"page {page_number}", font=FONT_MD, fill=255)
    draw.line([(2, 16), (OLED_WIDTH - 3, 16)], fill=255, width=1)
    y = 19
    for line in body[:5]:
        draw.text((2, y), line, font=FONT_SM, fill=255)
        y += FONT_SM.size + 1
    with _lock:
        _flush(img)


def show_vocab(word: str, definition: str) -> None:
    img = _blank()
    draw = ImageDraw.Draw(img)
    draw.text((2, 2), word.upper(), font=FONT_LG, fill=255)
    draw.line([(2, 20), (OLED_WIDTH - 3, 20)], fill=255, width=1)
    y = 24
    for line in wrap(definition, FONT_SM, OLED_WIDTH - 4)[:4]:
        draw.text((2, y), line, font=FONT_SM, fill=255)
        y += FONT_SM.size + 1
    with _lock:
        _flush(img)


def show_character(name: str, blurb: str) -> None:
    img = _blank()
    draw = ImageDraw.Draw(img)
    draw.text((2, 2), name, font=FONT_LG, fill=255)
    draw.line([(2, 20), (OLED_WIDTH - 3, 20)], fill=255, width=1)
    y = 24
    for line in wrap(blurb, FONT_SM, OLED_WIDTH - 4)[:4]:
        draw.text((2, y), line, font=FONT_SM, fill=255)
        y += FONT_SM.size + 1
    with _lock:
        _flush(img)


def show_answer(answer: str, refused: bool = False) -> None:
    img = _blank()
    draw = ImageDraw.Draw(img)
    header = "not yet" if refused else "answer"
    draw.text((2, 2), header, font=FONT_MD, fill=255)
    draw.line([(2, 16), (OLED_WIDTH - 3, 16)], fill=255, width=1)
    y = 19
    for line in wrap(answer, FONT_SM, OLED_WIDTH - 4)[:5]:
        draw.text((2, y), line, font=FONT_SM, fill=255)
        y += FONT_SM.size + 1
    with _lock:
        _flush(img)


def _render_qr(url: str, target_px: int) -> Image.Image:
    """Render a QR for `url` at roughly `target_px` on a side, using an
    integer box_size (pixels per module) so modules land on pixel boundaries.

    SSD1306 is 1-bit and tiny; a non-integer scale smears modules and makes
    the code unscannable. We render the QR at its native size (no resize)
    and return a PIL "1"-mode image with white modules on a black quiet zone
    (matches OLED lit-pixels = module)."""
    import qrcode

    # Error correction L keeps the code as small as possible; phones handle
    # low-correction QRs fine when modules are crisp. Border of 2 gives the
    # required quiet zone without wasting precious pixels.
    border = 2
    # Pick the largest box_size that still fits in target_px. Start with a
    # throw-away version just to discover module_count.
    probe = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=1,
        border=border,
    )
    probe.add_data(url)
    probe.make(fit=True)
    modules = probe.modules_count  # e.g. 25 for version 2
    total_modules = modules + 2 * border
    box_size = max(1, target_px // total_modules)

    qr = qrcode.QRCode(
        version=probe.version,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=box_size,
        border=border,
    )
    qr.add_data(url)
    qr.make(fit=True)
    # Render in the classic orientation: black modules on a white quiet zone.
    # On the OLED this means the QR region lights up (white pixels = on) with
    # dark modules cut out — the pattern every phone scanner expects. Inverted
    # QRs (white modules on dark) work on iOS and modern Android but break on
    # a surprising number of third-party scanners, so we don't risk it.
    return qr.make_image(fill_color="black", back_color="white").convert("1")


def show_qr(url: str, caption: str | None = None) -> None:
    """Welcome/connect screen: pixel-perfect QR on the left, 'scan to open
    Lumos' block on the right. The `caption` arg is accepted for backwards
    compatibility but ignored; the layout is fixed."""
    del caption  # layout is fixed; no dynamic caption

    qr_img = _render_qr(url, target_px=OLED_HEIGHT)  # ~58px for our URLs
    qw, qh = qr_img.size

    img = _blank()
    # Left: QR, vertically centered with a 1-px left margin.
    qx = 1
    qy = max(0, (OLED_HEIGHT - qh) // 2)
    img.paste(qr_img, (qx, qy))

    draw = ImageDraw.Draw(img)

    # Right column: from just past the QR to the right edge.
    col_x = qx + qw + 5
    col_w = OLED_WIDTH - col_x - 1

    def _centered(text: str, y: int, font) -> None:
        tw = _text_width(draw, text, font)
        x = col_x + max(0, (col_w - tw) // 2)
        draw.text((x, y), text, font=font, fill=255)

    # Three-line typographic block (QR already carries the URL, so no
    # redundant host footer that would just truncate on a 128px display):
    #   SCAN     (FONT_LG, 16px tall)
    #   to open  (FONT_SM, 10px)
    #   Lumos    (FONT_LG, 16px tall)
    _centered("SCAN",    2, FONT_LG)
    _centered("to open", 24, FONT_SM)
    _centered("Lumos",  40, FONT_LG)

    with _lock:
        _flush(img)


def snapshot_png() -> bytes:
    """Return a PNG of the last-rendered frame (for debugging without hardware)."""
    buf = io.BytesIO()
    _blank().save(buf, format="PNG")
    return buf.getvalue()


# ----- smoke test ----------------------------------------------------------

if __name__ == "__main__":
    import sys
    import time

    _init_oled()
    print("real oled:", is_real())
    show_status("Lumos", ["ready", "smoke test"])
    time.sleep(1.5)
    show_page_summary(
        312,
        "Smerdyakov lurks in the courtyard; Ivan's anxiety sharpens as the evening cools.",
    )
    time.sleep(1.5)
    show_vocab("perspicacious", "shrewdly discerning; notably perceptive")
    time.sleep(1.5)
    show_answer("Smerdyakov is a servant at the Karamazov estate introduced on p. 94.")
    time.sleep(1.5)
    show_qr("http://lumos.local:8080/library")
    time.sleep(1.5)
    show_ready("The Brothers Karamazov", 312)
    print("display smoke test sequence complete")
    sys.exit(0)
