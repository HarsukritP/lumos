# Lumos — reading light, with intelligence

Clip-on reading light that knows what you're reading. Camera watches the
page, OLED shows contextual info, button + mic let you ask questions aloud.
Runs entirely on a Pi Zero 2 W. Reading history never leaves the device;
only questions go to Gemini.

## Run it

```bash
cd /home/pi/lumos
python3 main.py
```

Boots in ~20 s on Pi Zero 2 W. You'll see "Lumos ready" on the OLED.
API + PWA served at http://lumos.local:8080/ .

Stop with Ctrl-C.

## Layout

- `config.py` — constants + env loading
- `db.py` — SQLite (books, pages, questions)
- `display.py` — OLED (SSD1306 via I2C)
- `camera.py` — rpicam-still wrapper + NCC similarity
- `audio.py` — arecord wrapper for the INMP441 I2S mic
- `ai.py` — Gemini 2.5 Flash (identify / summarize / answer / transcribe)
- `main.py` — orchestrator (watch loop + idle loop + button + Flask thread)
- `app/server.py` — Flask API + static PWA serving
- `app/static/dist/` — the built PWA (vanilla ES module SPA + service worker)

## Runtime shape

Three threads in one Python process:

- **watch** captures frames every 2 s, detects a stable page turn (NCC
  similarity, 3 s stability), then sends that frame to Gemini for
  identification (first frame) and page summary (every frame).
- **idle** rotates OLED cards (vocab / character / QR / status) every 8 s
  once there's been 20 s of no activity. QR pointing at the library page
  rotates in roughly every minute.
- **flask** serves the PWA + 7 read-only API endpoints.

Button on GPIO 17 triggers `handle_question()` in a worker thread:
record 5 s → transcribe → capture fresh page → answer (spoiler-safe) → render.

## API

- `GET /api/status` — current book / page / busy
- `GET /api/books` — library
- `GET /api/books/<id>` — book detail (with pages + questions inline)
- `GET /api/books/<id>/pages`
- `GET /api/books/<id>/questions`
- `GET /api/questions/<id>`
- `GET /api/vocab` — every vocab word, grouped by book

## Env

`~/.lumos.env` with `GEMINI_API_KEY=...`, chmod 600.

## Hardware

- Pi Zero 2 W (Raspberry Pi OS Lite 64-bit, Python 3.13)
- SSD1306 128x64 I2C OLED at 0x3c (SDA=GPIO 2, SCL=GPIO 3)
- IMX708 camera module 3 (rotated 180 in software)
- INMP441 I2S mic (SCK=18, WS=19, SD=20, L/R→GND)
- Push button on GPIO 17 (internal pull-up to 3V3)

System config needed once in `/boot/firmware/config.txt` `[all]` section:

```
dtparam=i2c_arm=on
dtparam=i2s=on
dtoverlay=googlevoicehat-soundcard
```
