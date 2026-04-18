"""Audio capture for the INMP441 I2S mic via `arecord`."""
from __future__ import annotations

import subprocess
from pathlib import Path

from config import (
    ARECORD_CHANNELS,
    ARECORD_DEVICE,
    ARECORD_FORMAT,
    ARECORD_RATE,
    QUESTION_RECORD_SECONDS,
    QUESTION_WAV,
)


class AudioError(RuntimeError):
    pass


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


# ----- smoke test ----------------------------------------------------------

if __name__ == "__main__":
    import sys
    import time

    print("Recording 3 seconds... speak into the mic now.")
    t0 = time.monotonic()
    path = record(seconds=3, out_path="/tmp/lumos/smoke_question.wav")
    dt = time.monotonic() - t0
    size = path.stat().st_size
    print(f"Recorded {size} bytes in {dt:.2f}s -> {path}")
    # Expected size at 48000Hz, 2ch, S32_LE, 3s: ~1.15MB + 44B header
    expected = 48000 * 2 * 4 * 3
    print(f"Expected ~{expected} bytes; got {size}.")
    sys.exit(0)
