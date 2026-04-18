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
    BOOK_SWITCH_HOLD_S,
    BOOK_SWITCH_SIMILARITY,
    BUTTON_PIN,
    CAPTURE_INTERVAL,
    FRAME_PATH,
    IDENTIFY_CONFIRMATIONS,
    IDENTIFY_MIN_CONFIDENCE,
    IDENTIFY_MIN_COVER_CONFIDENCE,
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

# STATE is deliberately imported from a dedicated module so the Flask server
# (which `from main import ...` via its own import of `main`) shares the
# exact same instance as this orchestrator. Running `python3 main.py` makes
# this file `__main__`, which Flask would otherwise re-import as `main`,
# creating a second State(). The state module dodges that.
from state import (
    PHASE_CONNECT,
    PHASE_HUNTING,
    PHASE_READING,
    STATE,
    note_remote_client,  # noqa: F401  (re-exported for app/server.py)
)

# How long to wait in connect with no remote client before auto-advancing
# (so the device is still demo-able without a phone, and so localhost-only
# testing isn't blocked).
CONNECT_AUTO_ADVANCE_S = 90.0


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


# ----- identify + commit ---------------------------------------------------

def _record_identify_attempt(title: str, author: str, conf: float) -> None:
    """Append to the rolling identify trail, keep only the last 8 entries."""
    tk, ak = db.normalize_identity(title, author)
    with STATE.lock:
        STATE.identify_trail.append((tk, ak, conf, time.time()))
        del STATE.identify_trail[:-8]


def _agreeing_identifies(window_s: float = 30.0) -> tuple[str, str, int] | None:
    """Look at recent identify attempts and, if the last N (>= IDENTIFY_CONFIRMATIONS)
    all agree on the same non-empty (title_key, author_key), return that
    key plus the agreement count. Otherwise None."""
    now = time.time()
    trail = [e for e in STATE.identify_trail if now - e[3] <= window_s and e[0]]
    if len(trail) < IDENTIFY_CONFIRMATIONS:
        return None
    last = trail[-IDENTIFY_CONFIRMATIONS:]
    tk, ak, _, _ = last[0]
    if not tk or tk == "unknown":
        return None
    for e in last[1:]:
        if e[0] != tk or e[1] != ak:
            return None
    return tk, ak, len(last)


def _try_identify_book(img: Image.Image, img_path: Path) -> bool:
    """Run identify_book on this frame; commit to a book only after the last
    IDENTIFY_CONFIRMATIONS attempts agree. Returns True iff a book was
    committed in this call (caller can then fall through to page-reading)."""
    display.show_status("Lumos", ["identifying", "this book..."])
    try:
        ident = ai.identify_book(img_path)
    except ai.AIError as e:
        log.warning("identify unavailable: %r", e)
        return False

    title = ident["title"]
    author = ident["author"]
    is_textbook = ident["is_textbook"]
    conf = ident["confidence"]
    cover = ident.get("cover_visible", False)
    oled_title = ident.get("oled_title", title)
    log.info(
        "identify: %r by %r tb=%s conf=%.2f cover=%s",
        title, author, is_textbook, conf, cover,
    )

    # Single-frame rejection gates (don't even add to the trail if we're
    # clearly looking at nothing). Cover frames are held to a higher bar.
    min_conf = IDENTIFY_MIN_COVER_CONFIDENCE if cover else IDENTIFY_MIN_CONFIDENCE
    if (
        conf < min_conf
        or (title.strip().lower() == "unknown" and author.strip().lower() == "unknown")
    ):
        log.info("identify rejected: conf %.2f < %.2f or unknown", conf, min_conf)
        with STATE.lock:
            STATE.last_committed = None
        return False

    _record_identify_attempt(title, author, conf)
    agree = _agreeing_identifies()
    if agree is None:
        n = sum(1 for e in STATE.identify_trail
                if db.normalize_identity(title, author)[0] == e[0])
        log.info("identify held: need %d agreeing, have %d of %r",
                 IDENTIFY_CONFIRMATIONS, n, title)
        # Reflect progress on the OLED so the demo feels alive.
        display.show_status(
            "Lumos",
            [f"saw \"{_clip(oled_title, 10)}\"", f"{n}/{IDENTIFY_CONFIRMATIONS} confirmed"],
        )
        return False

    tk, ak, count = agree
    row = db.find_or_create_book_by_identity(
        title, author, is_textbook, cover_phash=camera.phash(img),
    )
    with STATE.lock:
        STATE.book_id = row["id"]
        STATE.book_title = row["title"]
        STATE.book_author = row["author"]
        STATE.current_page = int(row.get("current_page") or 0)
        STATE.current_title_key = tk
        STATE.current_author_key = ak
        STATE.oled_title = oled_title
    STATE.set_phase(PHASE_READING, reason=f"identified {row['title']!r} ({count}/{IDENTIFY_CONFIRMATIONS})")
    return True


def _clip(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1] + "\u2026"


def _commit_page_read(img: Image.Image, img_path: Path) -> None:
    """Summarize the current page and persist it. Assumes a book is already
    identified and we're in PHASE_READING."""
    try:
        display.show_status("Lumos", ["reading", "the page..."])
        summary = ai.summarize_page(img_path, STATE.book_title, STATE.current_page)
        page_number = summary["page_number"]
        oled_summary = summary["oled_summary"]
        db.add_page(
            STATE.book_id,
            page_number,
            summary["summary"],
            oled_summary,
            summary["characters"],
            summary["vocabulary"],
            summary["concepts"],
        )
        db.set_current_page(STATE.book_id, page_number)

        with STATE.lock:
            STATE.current_page = page_number
            STATE.last_commit_at = time.time()

        for v in summary["vocabulary"]:
            if isinstance(v, dict) and v.get("word"):
                # Prefer the OLED-short form; fall back to the long one.
                definition = v.get("oled_definition") or v.get("definition") or ""
                if definition:
                    queue_vocab_card(v["word"], definition)
        for c in summary["characters"]:
            if isinstance(c, dict) and c.get("name"):
                role = c.get("oled_role") or c.get("role") or ""
                if role:
                    queue_character_card(c["name"], role)

        # Prefer the OLED-sized summary for the tiny screen.
        display.show_page_summary(page_number, oled_summary or summary["summary"])
        STATE.touch()
    except ai.AIError as e:
        log.warning("Gemini unavailable during commit: %r", e)
        display.show_status("Lumos", ["no signal,", "holding on to", "last page"])
    except Exception as e:
        log.exception("commit failed: %r", e)
        display.show_status("Lumos", ["hiccup", "try again"])


def _maybe_switch_book(img: Image.Image, img_path: Path) -> bool:
    """Detect that the reader swapped to a different physical book mid-session.

    Triggered only while we're in PHASE_READING and a frame's similarity to
    the last committed frame has been below BOOK_SWITCH_SIMILARITY for at
    least BOOK_SWITCH_HOLD_S seconds. Runs a fresh identify; if it reports a
    different (title_key, author_key), we drop the current book context and
    return True so the caller skips further processing this tick.
    """
    if STATE.last_committed is None:
        return False
    sim = camera.similarity(img, STATE.last_committed)
    now = time.time()
    if sim >= BOOK_SWITCH_SIMILARITY:
        # content close enough to what we committed — not a new book
        if STATE.dissimilar_since is not None:
            with STATE.lock:
                STATE.dissimilar_since = None
        return False
    if STATE.dissimilar_since is None:
        with STATE.lock:
            STATE.dissimilar_since = now
        return False
    if now - STATE.dissimilar_since < BOOK_SWITCH_HOLD_S:
        return False
    # Held dissimilar long enough — confirm with an identify.
    try:
        ident = ai.identify_book(img_path)
    except ai.AIError as e:
        log.warning("book-switch identify failed: %r", e)
        with STATE.lock:
            STATE.dissimilar_since = None
        return False
    new_tk, new_ak = db.normalize_identity(ident["title"], ident["author"])
    if not new_tk or new_tk == "unknown":
        # Still nothing clear under the lamp. Reset timer; stay in reading.
        with STATE.lock:
            STATE.dissimilar_since = None
        return False
    if new_tk == STATE.current_title_key and new_ak == STATE.current_author_key:
        # Same book, reader just flipped to a very different-looking page.
        with STATE.lock:
            STATE.dissimilar_since = None
        return False
    log.info(
        "book-switch: %r -> %r (sim=%.2f held %.1fs)",
        STATE.book_title, ident["title"], sim, now - STATE.dissimilar_since,
    )
    STATE.reset_book()
    STATE.set_phase(PHASE_HUNTING, reason="book changed")
    return True


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
            var, mean = camera.page_score(img)
            ok, reason = camera.is_likely_page(img)
            with STATE.lock:
                STATE.last_capture_at = time.time()
                STATE.last_capture_var = var
                STATE.last_capture_mean = mean
                STATE.last_capture_ok = ok
                STATE.last_capture_reason = reason

            # In connect phase, we still capture (so the debug view is live),
            # but we don't try to commit / identify anything.
            if STATE.phase == PHASE_CONNECT:
                dt = time.monotonic() - t0
                time.sleep(max(0.0, CAPTURE_INTERVAL - dt))
                continue

            if not ok:
                # Not book-like; don't let it become pending, don't call Gemini.
                log.debug("skipping frame: %s", reason)
                # If we had a pending candidate waiting to stabilize, drop it.
                if STATE.pending_frame is not None:
                    with STATE.lock:
                        STATE.pending_frame = None
                        STATE.pending_path = None
                        STATE.pending_since = None
                dt = time.monotonic() - t0
                time.sleep(max(0.0, CAPTURE_INTERVAL - dt))
                continue

            # While reading, check for a book-switch BEFORE running stability
            # logic — if the reader swapped books, we reset to hunting and
            # wait for fresh identify rather than accidentally summarizing
            # the new book's pages under the old book's row.
            if STATE.phase == PHASE_READING and not STATE.busy:
                with STATE.lock:
                    STATE.busy = True
                try:
                    if _maybe_switch_book(img, Path(str(path))):
                        # Reset to hunting; the next tick will start over.
                        continue
                finally:
                    with STATE.lock:
                        STATE.busy = False

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
                            if STATE.book_id is None:
                                # Strict scan flow: require N-in-a-row agreeing
                                # identifies before committing a book. On the
                                # commit call itself we also summarize this page.
                                committed_ok = _try_identify_book(
                                    committed, committed_path or Path(path)
                                )
                                if committed_ok:
                                    _commit_page_read(
                                        committed, committed_path or Path(path)
                                    )
                            else:
                                _commit_page_read(
                                    committed, committed_path or Path(path)
                                )
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

# How long after the last commit we stay on the "page summary" display before
# flipping to the caught-up card. Keeps the summary readable for a moment.
CAUGHT_UP_AFTER_COMMIT_S = 8.0

# Interval between re-flushing the same welcome/hunting screen so the OLED
# doesn't drift into burn-in on a single static frame during long demos.
STATIC_REFRESH_S = 20.0


def idle_loop() -> None:
    """Drives the OLED while the watch loop isn't actively committing.

    Phase drives the primary display:
      connect  -> persistent welcome QR ("scan me")
      hunting  -> "looking for a book" card
      reading  -> big "p. N  ·  caught up" card, occasionally rotating to a
                  queued vocab/character card for ~8s then returning.
    """
    log.info("idle loop starting")
    last_static_flush = 0.0
    last_rotation_at = 0.0
    rotation_showing_card = False

    while not STATE.shutdown:
        try:
            now = time.time()

            # Demo-friendly fallback: if nobody scanned the QR after a while,
            # advance anyway so local / on-device testing can proceed.
            if (
                STATE.phase == PHASE_CONNECT
                and now - STATE.started_at >= CONNECT_AUTO_ADVANCE_S
            ):
                STATE.set_phase(PHASE_HUNTING, reason="connect auto-advance")

            # Don't fight the watch loop / question handler for the OLED.
            if STATE.busy:
                time.sleep(0.5)
                continue

            if STATE.phase == PHASE_CONNECT:
                if now - last_static_flush >= STATIC_REFRESH_S or last_static_flush == 0.0:
                    display.show_qr(LIBRARY_URL)
                    last_static_flush = now
                    STATE.last_qr_at = now
                time.sleep(1.0)
                continue

            if STATE.phase == PHASE_HUNTING:
                if now - last_static_flush >= STATIC_REFRESH_S or last_static_flush == 0.0:
                    display.show_status("Lumos", ["looking for", "a book..."])
                    last_static_flush = now
                time.sleep(1.0)
                continue

            # PHASE_READING
            since_commit = now - (STATE.last_commit_at or 0)
            if since_commit < CAUGHT_UP_AFTER_COMMIT_S:
                # Commit just happened; let show_page_summary linger.
                time.sleep(0.5)
                continue

            # Primary display: big page number, "caught up" footer.
            # Every IDLE_CARD_SECONDS, rotate to a vocab/character card for
            # one beat then return to caught_up. If no queue, stay on caught_up.
            with STATE.lock:
                cards = list(STATE.idle_cards)

            if cards and now - last_rotation_at >= IDLE_CARD_SECONDS:
                idx = STATE.current_card_index % len(cards)
                render_idle_card(cards[idx])
                STATE.current_card_index = (idx + 1) % len(cards)
                last_rotation_at = now
                rotation_showing_card = True
                last_static_flush = 0.0
                time.sleep(IDLE_CARD_SECONDS)
                continue

            # Re-flush caught_up periodically so we recover from any transient
            # I2C glitch or screen drift.
            if rotation_showing_card or now - last_static_flush >= STATIC_REFRESH_S:
                display.show_caught_up(STATE.current_page, STATE.oled_title or STATE.book_title)
                last_static_flush = now
                rotation_showing_card = False
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
            result["oled_answer"],
            result["refused_as_spoiler"],
        )
        display.show_answer(
            result["oled_answer"] or result["answer"],
            refused=result["refused_as_spoiler"],
        )
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

    # connect phase: show welcome QR immediately. Idle loop keeps it refreshed.
    display.show_qr(LIBRARY_URL)
    log.info("Lumos ready (phase=%s). Ctrl-C to stop.", STATE.phase)
    try:
        while not STATE.shutdown:
            time.sleep(1.0)
    except KeyboardInterrupt:
        _clean_shutdown()
    log.info("bye.")


if __name__ == "__main__":
    main()
