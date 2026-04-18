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

BUTTON_PIN = 17

OLED_WIDTH = 128
OLED_HEIGHT = 64
OLED_I2C_ADDR = 0x3C

CAPTURE_INTERVAL = 2.0
PAGE_STABILITY_TIME = 3.0
SIMILARITY_THRESHOLD = 0.75

IDLE_TIMEOUT = 20.0
IDLE_CARD_SECONDS = 8.0
QR_EVERY_SECONDS = 60.0

QUESTION_RECORD_SECONDS = 5
ARECORD_DEVICE = "plughw:0,0"
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
