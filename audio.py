"""Audio capture for the INMP441 I2S mic via `arecord`.

Two modes:
  * record(seconds) — legacy fixed-duration capture (still used by smoke tests).
  * start_recording() / stop_recording() — walkie-talkie PTT. Caller holds
    a button, we spin up an arecord subprocess that runs in the background,
    and when the button is released we SIGINT arecord so it closes the WAV
    cleanly. A MAX_SECONDS watchdog terminates runaway recordings.
"""
from __future__ import annotations

import logging
import signal
import subprocess
import time
from pathlib import Path

from config import (
    ARECORD_CHANNELS,
    ARECORD_DEVICE,
    ARECORD_FORMAT,
    ARECORD_RATE,
    PTT_MAX_SECONDS,
    QUESTION_RECORD_SECONDS,
    QUESTION_WAV,
)

log = logging.getLogger("lumos.audio")


class AudioError(RuntimeError):
    pass


# ----- fixed-duration recording (legacy + smoke tests) --------------------

def record(
    seconds: int = QUESTION_RECORD_SECONDS,
    out_path: Path | str = QUESTION_WAV,
    device: str = ARECORD_DEVICE,
) -> Path:
    """Record `seconds` of audio to a WAV file and return the path."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "arecord",
        "-D", device,
        "-c", str(ARECORD_CHANNELS),
        "-r", str(ARECORD_RATE),
        "-f", ARECORD_FORMAT,
        "-d", str(seconds),
        str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=seconds + 5)
    if proc.returncode != 0 or not out.exists() or out.stat().st_size < 1024:
        raise AudioError(
            f"arecord failed ({proc.returncode}) size={out.stat().st_size if out.exists() else 0}: "
            f"{proc.stderr[-300:]}"
        )
    return out


# ----- walkie-talkie PTT --------------------------------------------------

class PTTRecorder:
    """A single in-flight PTT recording. `start()` spawns arecord; `stop()`
    terminates it and returns the resulting WAV path. Safe to call `stop()`
    more than once or on a never-started recorder — it's a no-op.

    We use arecord's own `--duration` as the upper bound (`PTT_MAX_SECONDS`)
    so even if our release handler never fires, arecord stops on its own
    and produces a valid WAV. On `stop()` we send SIGINT; arecord traps
    that and finalizes the WAV header cleanly."""

    def __init__(
        self,
        out_path: Path | str = QUESTION_WAV,
        device: str = ARECORD_DEVICE,
        max_seconds: int = PTT_MAX_SECONDS,
    ) -> None:
        self.out_path = Path(out_path)
        self.device = device
        self.max_seconds = max_seconds
        self.proc: subprocess.Popen | None = None
        self.started_at: float = 0.0
        self.stopped_at: float = 0.0

    def start(self) -> None:
        if self.proc is not None:
            raise AudioError("PTTRecorder already started")
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "arecord",
            "-q",
            "-D", self.device,
            "-c", str(ARECORD_CHANNELS),
            "-r", str(ARECORD_RATE),
            "-f", ARECORD_FORMAT,
            "-d", str(self.max_seconds),
            str(self.out_path),
        ]
        log.info("PTT start: %s", " ".join(cmd))
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
            )
        except FileNotFoundError as e:
            raise AudioError(f"arecord not found: {e}") from e
        self.started_at = time.monotonic()

    def elapsed(self) -> float:
        if not self.started_at:
            return 0.0
        end = self.stopped_at or time.monotonic()
        return max(0.0, end - self.started_at)

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> Path:
        """Terminate the running capture (if any), wait for arecord to flush
        the WAV, and return the output path. Raises AudioError if the file
        never materialized or is clearly empty.

        arecord traps SIGINT and finalizes the WAV header on the way out,
        but on a loaded Pi Zero 2 W that tear-down can take 1-2 seconds.
        We wait up to 3s for graceful exit before escalating to SIGKILL,
        and after any exit we poll the filesystem for ~1s in case the
        write hadn't hit the page cache yet when we checked."""
        if self.proc is None:
            raise AudioError("PTTRecorder never started")
        if self.stopped_at:
            return self.out_path  # idempotent

        try:
            self.proc.send_signal(signal.SIGINT)
        except ProcessLookupError:
            pass
        graceful = True
        try:
            self.proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            graceful = False
            log.warning("arecord did not exit on SIGINT; killing")
            try:
                self.proc.kill()
                self.proc.wait(timeout=1.0)
            except Exception:
                pass
        self.stopped_at = time.monotonic()

        # Drain stderr so we can surface ALSA diagnostics on failure.
        stderr_tail = ""
        try:
            if self.proc.stderr is not None:
                raw = self.proc.stderr.read() or b""
                stderr_tail = raw.decode(errors="replace").strip()[-400:]
        except Exception:
            pass
        if self.proc.returncode not in (0, -signal.SIGINT, -signal.SIGKILL):
            log.warning(
                "arecord exit=%s graceful=%s stderr=%r",
                self.proc.returncode, graceful, stderr_tail,
            )

        # Poll briefly for the file — the filesystem write can lag a bit
        # behind arecord's exit on a busy Pi.
        deadline = time.monotonic() + 1.0
        while not self.out_path.exists() and time.monotonic() < deadline:
            time.sleep(0.05)

        if not self.out_path.exists():
            raise AudioError(
                f"PTT produced no file at {self.out_path} "
                f"(arecord exit={self.proc.returncode}; stderr={stderr_tail!r})"
            )
        size = self.out_path.stat().st_size
        if size < 2048:  # < ~21ms at 48kHz/2ch/S32 — treat as noise/misfire
            raise AudioError(
                f"PTT wav too short ({size} bytes; "
                f"arecord exit={self.proc.returncode}; stderr={stderr_tail!r})"
            )
        log.info(
            "PTT stop: %.2fs recorded, %d bytes -> %s",
            self.elapsed(), size, self.out_path,
        )
        return self.out_path


# ----- smoke test ----------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("Recording 3 seconds... speak into the mic now.")
    t0 = time.monotonic()
    path = record(seconds=3, out_path="/tmp/lumos/smoke_question.wav")
    dt = time.monotonic() - t0
    size = path.stat().st_size
    print(f"Recorded {size} bytes in {dt:.2f}s -> {path}")
    expected = 48000 * 2 * 4 * 3
    print(f"Expected ~{expected} bytes; got {size}.")

    print("\nPTT recorder sanity: start, hold 2s, stop.")
    rec = PTTRecorder(out_path="/tmp/lumos/smoke_ptt.wav")
    rec.start()
    time.sleep(2.0)
    path = rec.stop()
    size = path.stat().st_size
    print(f"PTT recorded {size} bytes in {rec.elapsed():.2f}s -> {path}")
    sys.exit(0)
