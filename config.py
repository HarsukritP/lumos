"""Lumos runtime configuration. All constants live here; no state."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values


HOME = Path.home()
PROJECT_DIR = Path(__file__).resolve().parent
# Primary location: a .env file inside the project; fall back to the legacy
# ~/.lumos.env location for anyone who hasn't migrated yet.
_PROJECT_ENV = PROJECT_DIR / ".env"
_LEGACY_ENV = HOME / ".lumos.env"
ENV_PATH = _PROJECT_ENV if _PROJECT_ENV.exists() else _LEGACY_ENV
DB_PATH = HOME / "lumos.db"
TMP_DIR = Path("/tmp/lumos")
TMP_DIR.mkdir(parents=True, exist_ok=True)

FRAME_PATH = TMP_DIR / "frame.jpg"
PENDING_PATH = TMP_DIR / "pending.jpg"
QUESTION_WAV = TMP_DIR / "question.wav"

CAMERA_ROTATION = 180
CAMERA_TIMEOUT_MS = 500
CAPTURE_WIDTH = 800
CAPTURE_HEIGHT = 600

BUTTON_PIN = 17

OLED_WIDTH = 128
OLED_HEIGHT = 64
OLED_I2C_ADDR = 0x3C

CAPTURE_INTERVAL = 5.0
PAGE_STABILITY_TIME = 3.0
SIMILARITY_THRESHOLD = 0.75
# Post-stability dedup: if a candidate commit's similarity to last_committed
# is at or above this, treat it as the same page and DO NOT call Gemini to
# summarize it again. Empirically a real page turn drops sim well below 0.75;
# 0.88 keeps us safe against lighting drift / minor hand shadows.
DEDUP_SIMILARITY = 0.88

# Local "does this frame look like a book page/cover?" gate. Runs before we
# ever call Gemini, so pointing the camera at a ceiling / desk / shadow never
# burns API quota. Tuned empirically on the rpicam-still output.
PAGE_SCORE_MIN = 15.0         # Laplacian variance on a 128x128 grayscale
PAGE_BRIGHTNESS_MIN = 40.0    # 0..255, reject very dark frames (mean)
PAGE_BRIGHTNESS_MAX = 235.0   # reject blown-out white frames
# Fraction of pixels with brightness > 160 (paper detection). Book pages
# under a reading lamp have >0.35; covers >0.20; a random desk/room <0.16.
PAGE_BRIGHT_FRAC_MIN = 0.20

# If Gemini's book-identification confidence is below this, don't commit a
# book row or summarize — we're probably looking at nothing/hands/noise.
IDENTIFY_MIN_CONFIDENCE = 0.35

# Strict book-scan flow. We require N consecutive identifies to agree on a
# normalized (title, author) before we commit a book. This stops a single
# blurry frame ("is that Dune?") from creating a book row.
IDENTIFY_CONFIRMATIONS = 2            # how many agreeing identifies needed
IDENTIFY_MIN_COVER_CONFIDENCE = 0.55  # higher bar for cover-only IDs

# Book-switch detection. While reading, if a new frame's similarity to the
# last committed frame drops below this for BOOK_SWITCH_HOLD_S seconds AND a
# re-identify returns a different book, we jump back to HUNTING. Deliberately
# lower than SIMILARITY_THRESHOLD (0.75) so normal page turns don't trigger.
BOOK_SWITCH_SIMILARITY = 0.35
BOOK_SWITCH_HOLD_S = 4.0

# "Resume from last page" prompt timeout. If the reader hasn't flipped to
# (or past) their last-known page within this many seconds after a re-
# identify, auto-dismiss the prompt and start summarizing from whatever
# page we're currently looking at.
RESUME_TIMEOUT_S = 90.0

# Walkie-talkie push-to-talk. Hold-to-record; release = stop + process.
# If the user holds past this, we auto-stop so arecord doesn't run forever.
PTT_MAX_SECONDS = 10
# Minimum hold needed before we consider it a real recording (protects
# against stray short presses getting transcribed as noise).
PTT_MIN_SECONDS = 0.4
# Long-press threshold used INSIDE the resume prompt to "skip resume".
# Unrelated to PTT: if the reader holds the button >= this while
# awaiting_resume, we drop the prompt and accept the current frame as the
# starting page.
RESUME_SKIP_HOLD_S = 2.0

IDLE_TIMEOUT = 20.0
IDLE_CARD_SECONDS = 8.0
QR_EVERY_SECONDS = 60.0
# How long the OLED answer stays on screen after a PTT question is
# answered. During this window STATE.busy remains True so the idle
# loop cannot overwrite it.
ANSWER_DISPLAY_S = 8.0

QUESTION_RECORD_SECONDS = 5
# The INMP441 is a mono I2S mic on the LEFT channel. We record in the
# hardware's native format (S32_LE stereo 48kHz) then post-process to
# mono 16kHz 16-bit in audio.py before sending to Gemini. The ALSA
# plug layer was silently dropping data when asked to convert inline.
ARECORD_DEVICE = "hw:0,0"
ARECORD_RATE = 48000
ARECORD_CHANNELS = 2
ARECORD_FORMAT = "S32_LE"

MODEL_NAME = "gemini-2.5-flash"

FLASK_HOST = "0.0.0.0"
FLASK_PORT = 8080
LIBRARY_URL = f"http://lumos.local:{FLASK_PORT}/library"

STATIC_DIR = PROJECT_DIR / "app" / "static" / "dist"


def _load_env() -> dict:
    if not ENV_PATH.exists():
        return {}
    return dict(dotenv_values(str(ENV_PATH)))


_env = _load_env()
GEMINI_API_KEY = _env.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")


def has_api_key() -> bool:
    return bool(GEMINI_API_KEY)
