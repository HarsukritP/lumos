"""Lumos orchestrator.

Three threads:
  - watch_loop: capture frames, detect page changes, commit summaries
  - idle_loop:  rotate OLED cards (vocab / character / QR / status) when idle
  - flask_thread: serve the PWA + API (launched from app.server)
Main thread holds signal handlers and sleeps.

Button handler (gpiozero runs its own thread) launches handle_question() to
record -> transcribe -> answer -> render.
"""
from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

from PIL import Image

import ai
import audio
import camera
import db
import display
from config import (
    BUTTON_PIN,
    CAPTURE_INTERVAL,
    FRAME_PATH,
    IDLE_CARD_SECONDS,
    IDLE_TIMEOUT,
    LIBRARY_URL,
    PAGE_STABILITY_TIME,
    PENDING_PATH,
    QR_EVERY_SECONDS,
    QUESTION_RECORD_SECONDS,
    QUESTION_WAV,
    SIMILARITY_THRESHOLD,
    has_api_key,
)

log = logging.getLogger("lumos.main")


# ----- shared state --------------------------------------------------------

class State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        # book identity
        self.book_id: int | None = None
        self.book_title: str = "Unknown"
        self.book_author: str = "Unknown"
        self.current_page: int = 0
        # frame comparison
        self.last_committed: Image.Image | None = None
        self.pending_frame: Image.Image | None = None
        self.pending_path: Path | None = None
        self.pending_since: float | None = None
        # activity / busy
        self.last_activity: float = time.time()
        self.busy: bool = False
        # idle card rotation
        self.idle_cards: list[dict[str, Any]] = []
        self.current_card_index: int = 0
        self.last_card_at: float = 0.0
        self.last_qr_at: float = 0.0
        # boot/run metadata
        self.started_at: float = time.time()
        self.shutdown: bool = False

    def touch(self) -> None:
        self.last_activity = time.time()

    def to_status(self) -> dict:
        return {
            "book_id": self.book_id,
            "book_title": self.book_title,
            "book_author": self.book_author,
            "current_page": self.current_page,
            "busy": self.busy,
            "started_at": self.started_at,
            "uptime_s": time.time() - self.started_at,
        }


STATE = State()


# ----- idle card helpers ---------------------------------------------------

def queue_vocab_card(word: str, definition: str) -> None:
    with STATE.lock:
        # avoid duplicates for the same word
        for c in STATE.idle_cards:
            if c.get("type") == "vocab" and c.get("word", "").lower() == word.lower():
                return
        STATE.idle_cards.append({"type": "vocab", "word": word, "definition": definition})


def queue_character_card(name: str, role: str) -> None:
    with STATE.lock:
        for c in STATE.idle_cards:
            if c.get("type") == "character" and c.get("name") == name:
                return
        STATE.idle_cards.append({"type": "character", "name": name, "role": role})


def render_idle_card(card: dict) -> None:
    t = card.get("type")
    if t == "vocab":
        display.show_vocab(card["word"], card["definition"])
    elif t == "character":
        display.show_character(card["name"], card["role"])
    elif t == "qr":
        display.show_qr(LIBRARY_URL)
    elif t == "status":
        display.show_ready(
            STATE.book_title if STATE.book_id else None,
            STATE.current_page if STATE.book_id else None,
        )
    else:
        display.show_ready(STATE.book_title if STATE.book_id else None, STATE.current_page)


# ----- capture + commit ----------------------------------------------------

def _commit_page(img: Image.Image, img_path: Path) -> None:
    """Called when pending_frame has been stable long enough.

    If there's no current book, run identify_book first. Then always run
    summarize_page to persist what's on this page.
    """
    try:
        if STATE.book_id is None:
            display.show_status("Lumos", ["identifying", "this book..."])
            ident = ai.identify_book(img_path)
            title = ident["title"]
            author = ident["author"]
            is_textbook = ident["is_textbook"]
            log.info("identified: %r by %r (tb=%s, conf=%.2f)", title, author, is_textbook, ident["confidence"])
            phash = camera.phash(img)
            row = db.get_or_create_book(phash)
            db.update_book_identity(row["id"], title, author, is_textbook)
            with STATE.lock:
                STATE.book_id = row["id"]
                STATE.book_title = title
                STATE.book_author = author

        display.show_status("Lumos", ["reading", "the page..."])
        summary = ai.summarize_page(img_path, STATE.book_title, STATE.current_page)
        page_number = summary["page_number"]
        db.add_page(
            STATE.book_id,
            page_number,
            summary["summary"],
            summary["characters"],
            summary["vocabulary"],
            summary["concepts"],
        )
        db.set_current_page(STATE.book_id, page_number)

        with STATE.lock:
            STATE.current_page = page_number

        for v in summary["vocabulary"]:
            if isinstance(v, dict) and v.get("word") and v.get("definition"):
                queue_vocab_card(v["word"], v["definition"])
        for c in summary["characters"]:
            if isinstance(c, dict) and c.get("name") and c.get("role"):
                queue_character_card(c["name"], c["role"])

        display.show_page_summary(page_number, summary["summary"])
        STATE.touch()
    except ai.AIError as e:
        log.warning("Gemini unavailable during commit: %r", e)
        display.show_status("Lumos", ["no signal,", "holding on to", "last page"])
    except Exception as e:
        log.exception("commit failed: %r", e)
        display.show_status("Lumos", ["hiccup", "try again"])


def watch_loop() -> None:
    log.info("watch loop starting")
    while not STATE.shutdown:
        t0 = time.monotonic()
        try:
            if STATE.busy:
                time.sleep(0.2)
                continue
            path = camera.capture(FRAME_PATH)
            img = camera.load_oriented(path)

            if STATE.last_committed is None and STATE.pending_frame is None:
                # first frame ever: treat it as the start of a pending commit
                with STATE.lock:
                    STATE.pending_frame = img
                    STATE.pending_path = Path(str(path))
                    STATE.pending_since = time.time()
            else:
                ref = STATE.pending_frame or STATE.last_committed
                sim = camera.similarity(img, ref)

                if sim >= SIMILARITY_THRESHOLD:
                    # same as what we were watching
                    if (
                        STATE.pending_frame is not None
                        and STATE.pending_since is not None
                        and time.time() - STATE.pending_since >= PAGE_STABILITY_TIME
                    ):
                        committed = STATE.pending_frame
                        committed_path = STATE.pending_path
                        with STATE.lock:
                            STATE.last_committed = committed
                            STATE.pending_frame = None
                            STATE.pending_path = None
                            STATE.pending_since = None
                            STATE.busy = True
                        try:
                            _commit_page(committed, committed_path or Path(path))
                        finally:
                            with STATE.lock:
                                STATE.busy = False
                else:
                    # new content — save as pending
                    pending_copy = PENDING_PATH
                    try:
                        img.save(pending_copy, format="JPEG", quality=85)
                    except Exception:
                        pass
                    with STATE.lock:
                        STATE.pending_frame = img
                        STATE.pending_path = pending_copy
                        STATE.pending_since = time.time()
        except camera.CameraError as e:
            log.warning("camera error: %r", e)
            time.sleep(1.0)
        except Exception as e:
            log.exception("watch loop error: %r", e)
            time.sleep(0.5)

        dt = time.monotonic() - t0
        sleep_for = max(0.0, CAPTURE_INTERVAL - dt)
        time.sleep(sleep_for)


# ----- idle loop -----------------------------------------------------------

def idle_loop() -> None:
    log.info("idle loop starting")
    while not STATE.shutdown:
        try:
            now = time.time()
            idle_for = now - STATE.last_activity
            if idle_for < IDLE_TIMEOUT or STATE.busy:
                time.sleep(1.0)
                continue

            # Build rotation: queued cards + periodic QR + status fallback
            with STATE.lock:
                cards = list(STATE.idle_cards)
            if not cards:
                cards = [{"type": "status"}]

            # Inject QR card if we haven't shown one recently
            if now - STATE.last_qr_at > QR_EVERY_SECONDS:
                cards = cards + [{"type": "qr"}]
                STATE.last_qr_at = now

            if now - STATE.last_card_at >= IDLE_CARD_SECONDS:
                idx = STATE.current_card_index % len(cards)
                render_idle_card(cards[idx])
                STATE.current_card_index = (idx + 1) % len(cards)
                STATE.last_card_at = now

            time.sleep(1.0)
        except Exception as e:
            log.exception("idle loop error: %r", e)
            time.sleep(1.0)


# ----- button / question flow ---------------------------------------------

def handle_question() -> None:
    """Full button-press flow: record -> transcribe -> answer -> render."""
    if STATE.busy:
        display.show_status("Lumos", ["busy...", "one moment"])
        return
    with STATE.lock:
        STATE.busy = True
    STATE.touch()
    try:
        display.show_status("Lumos", [f"listening... ({QUESTION_RECORD_SECONDS}s)"])
        try:
            wav = audio.record(QUESTION_RECORD_SECONDS, QUESTION_WAV)
        except audio.AudioError as e:
            log.warning("audio record failed: %r", e)
            display.show_status("Lumos", ["mic error", "try again"])
            return

        display.show_status("Lumos", ["thinking..."])
        try:
            question = ai.transcribe_audio(wav)
        except ai.AIError as e:
            log.warning("transcribe failed: %r", e)
            display.show_status("Lumos", ["no signal,", "couldn't hear"])
            return

        if not question:
            display.show_status("Lumos", ["didn't catch", "that, again?"])
            return

        log.info("question: %r", question)

        try:
            path = camera.capture(FRAME_PATH)
        except camera.CameraError:
            path = None
        summaries = db.recent_summaries(STATE.book_id, 5) if STATE.book_id else []
        try:
            result = ai.answer_question(
                path, question, STATE.book_title, STATE.current_page, summaries
            )
        except ai.AIError as e:
            log.warning("answer failed: %r", e)
            display.show_status("Lumos", ["no signal,", "try again"])
            return

        db.add_question(
            STATE.book_id,
            STATE.current_page if STATE.book_id else None,
            question,
            result["answer"],
            result["refused_as_spoiler"],
        )
        display.show_answer(result["answer"], refused=result["refused_as_spoiler"])
    finally:
        with STATE.lock:
            STATE.busy = False
        STATE.touch()


def _on_button() -> None:
    threading.Thread(target=handle_question, daemon=True, name="button-handler").start()


def install_button() -> None:
    try:
        from gpiozero import Button
    except Exception as e:
        log.warning("gpiozero unavailable: %r — button disabled", e)
        return
    try:
        btn = Button(BUTTON_PIN, pull_up=True, bounce_time=0.05)
        btn.when_pressed = _on_button
        # Keep a module-level reference so it isn't GC'd
        globals()["_BUTTON"] = btn
        log.info("button installed on GPIO %d", BUTTON_PIN)
    except Exception as e:
        log.warning("button init failed: %r — button disabled", e)


# ----- shutdown ------------------------------------------------------------

def _clean_shutdown(signum=None, frame=None) -> None:
    log.info("shutdown signal %r", signum)
    STATE.shutdown = True
    try:
        display.clear()
    except Exception:
        pass


# ----- main ---------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # quiet noisy libs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("google_genai").setLevel(logging.WARNING)

    if not has_api_key():
        log.error("GEMINI_API_KEY not set. Check ~/.lumos.env.")
        display.show_status("Lumos", ["no API key", "check env"])
        sys.exit(2)

    db.init_db()
    display.show_status("Lumos", ["booting..."])

    signal.signal(signal.SIGINT, _clean_shutdown)
    signal.signal(signal.SIGTERM, _clean_shutdown)

    install_button()

    # Flask
    try:
        from app.server import start_in_thread
        start_in_thread()
    except Exception as e:
        log.warning("Flask server failed to start: %r", e)

    threading.Thread(target=watch_loop, daemon=True, name="watch").start()
    threading.Thread(target=idle_loop, daemon=True, name="idle").start()

    display.show_ready(None, None)
    log.info("Lumos ready. Ctrl-C to stop.")
    try:
        while not STATE.shutdown:
            time.sleep(1.0)
    except KeyboardInterrupt:
        _clean_shutdown()
    log.info("bye.")


if __name__ == "__main__":
    main()
