"""Shared Lumos runtime state.

Lives in its own module so that when `python3 main.py` runs the orchestrator
(as __main__) AND the Flask server's `from app.server import ...` triggers an
import of `main`, they don't each get their own State() object. Both sides
just `from state import STATE` and see the same instance.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


# ----- lifecycle phases ----------------------------------------------------
# connect: boot; OLED shows QR + "scan to begin". No Gemini calls yet.
# hunting: reader is connected; watch loop actively trying to identify a book.
# reading: book identified; tracking page changes. Idle display = big page #.
PHASE_CONNECT = "connect"
PHASE_HUNTING = "hunting"
PHASE_READING = "reading"


class State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        # lifecycle
        self.phase: str = PHASE_CONNECT
        # book identity
        self.book_id: int | None = None
        self.book_title: str = "Unknown"
        self.book_author: str = "Unknown"
        self.current_page: int = 0
        # frame comparison
        self.last_committed: "Image.Image | None" = None
        self.last_commit_at: float = 0.0
        self.pending_frame: "Image.Image | None" = None
        self.pending_path: Path | None = None
        self.pending_since: float | None = None
        # activity / busy
        self.last_activity: float = time.time()
        self.busy: bool = False
        # idle card rotation (secondary to caught-up display)
        self.idle_cards: list[dict[str, Any]] = []
        self.current_card_index: int = 0
        self.last_card_at: float = 0.0
        self.last_qr_at: float = 0.0
        # boot/run metadata
        self.started_at: float = time.time()
        self.shutdown: bool = False
        # debug / latest capture metadata (populated by watch loop)
        self.last_capture_at: float = 0.0
        self.last_capture_var: float = 0.0
        self.last_capture_mean: float = 0.0
        self.last_capture_bright_frac: float = 0.0
        self.last_capture_ok: bool = False
        self.last_capture_reason: str = ""
        self.last_capture_dt: float = 0.0

        # strict book-scan state
        # last N identify attempts as (title_key, author_key, confidence, ts)
        self.identify_trail: list[tuple[str, str, float, float]] = []
        self.current_title_key: str | None = None
        self.current_author_key: str | None = None
        self.oled_title: str = ""   # short title for the OLED (populated at identify)
        # book-switch tracking
        self.dissimilar_since: float | None = None

        # "Open a chat" resume flow. After identifying a book we've seen
        # before (current_page > 0 at identify time), we hold in
        # awaiting_resume until the reader flips to that page (or past it),
        # long-presses the button to skip, or the RESUME_TIMEOUT_S expires.
        # While awaiting_resume, the watch loop does NOT call summarize_page
        # and the OLED shows the resume prompt.
        self.awaiting_resume: bool = False
        self.resume_target_page: int = 0
        self.awaiting_resume_since: float = 0.0
        # Most recent printed page number OCR'd from a stable frame. None if
        # the last OCR failed or the frame had no detectable number.
        self.last_detected_page: int | None = None

    def touch(self) -> None:
        self.last_activity = time.time()

    def set_phase(self, new_phase: str, reason: str = "") -> None:
        import logging
        log = logging.getLogger("lumos.state")
        with self.lock:
            if self.phase == new_phase:
                return
            old = self.phase
            self.phase = new_phase
        log.info("phase %s -> %s%s", old, new_phase, f" ({reason})" if reason else "")

    def to_status(self) -> dict:
        return {
            "phase": self.phase,
            "book_id": self.book_id,
            "book_title": self.book_title,
            "book_author": self.book_author,
            "current_page": self.current_page,
            "busy": self.busy,
            "started_at": self.started_at,
            "uptime_s": time.time() - self.started_at,
            "last_commit_at": self.last_commit_at,
            "awaiting_resume": self.awaiting_resume,
            "resume_target_page": self.resume_target_page,
            "last_detected_page": self.last_detected_page,
        }

    def to_debug(self) -> dict:
        trail = [
            {"title_key": t, "author_key": a, "conf": c, "age_s": time.time() - ts}
            for (t, a, c, ts) in self.identify_trail[-5:]
        ]
        return {
            "phase": self.phase,
            "book_title": self.book_title,
            "oled_title": self.oled_title,
            "current_page": self.current_page,
            "busy": self.busy,
            "last_capture_at": self.last_capture_at,
            "last_capture_var": self.last_capture_var,
            "last_capture_mean": self.last_capture_mean,
            "last_capture_bright_frac": self.last_capture_bright_frac,
            "last_capture_ok": self.last_capture_ok,
            "last_capture_reason": self.last_capture_reason,
            "last_capture_dt": self.last_capture_dt,
            "last_commit_at": self.last_commit_at,
            "uptime_s": time.time() - self.started_at,
            "identify_trail": trail,
            "dissimilar_since": self.dissimilar_since,
            "awaiting_resume": self.awaiting_resume,
            "resume_target_page": self.resume_target_page,
            "last_detected_page": self.last_detected_page,
        }

    def reset_book(self) -> None:
        """Drop the current book context so the watch loop starts hunting again."""
        with self.lock:
            self.book_id = None
            self.book_title = "Unknown"
            self.book_author = "Unknown"
            self.current_page = 0
            self.last_committed = None
            self.pending_frame = None
            self.pending_path = None
            self.pending_since = None
            self.identify_trail.clear()
            self.current_title_key = None
            self.current_author_key = None
            self.oled_title = ""
            self.dissimilar_since = None
            self.awaiting_resume = False
            self.resume_target_page = 0
            self.awaiting_resume_since = 0.0
            self.last_detected_page = None


STATE = State()


def note_remote_client(remote_addr: str | None) -> None:
    """First non-loopback hit promotes us from connect -> hunting."""
    if not remote_addr:
        return
    if remote_addr in ("127.0.0.1", "::1", "localhost"):
        return
    if STATE.phase == PHASE_CONNECT:
        STATE.set_phase(PHASE_HUNTING, reason=f"remote client {remote_addr}")
