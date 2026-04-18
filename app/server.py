"""Flask API + static PWA server for Lumos.

Runs inside the same Python process as the orchestrator, on its own thread.
Read-only: there's no auth because there's no mutation surface exposed.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file, send_from_directory
from flask_cors import CORS

# Make sibling modules importable when Flask is started from the package
_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

import db
from config import FLASK_HOST, FLASK_PORT, FRAME_PATH, QUESTION_WAV, STATIC_DIR

log = logging.getLogger("lumos.server")

app = Flask(__name__, static_folder=None)
CORS(app)


@app.before_request
def _note_client():
    # First non-loopback hit on any endpoint advances the phase out of connect.
    try:
        from state import note_remote_client
        note_remote_client(request.remote_addr)
    except Exception:
        pass


# ----- API -----------------------------------------------------------------

@app.get("/api/status")
def api_status():
    try:
        from state import STATE
        return jsonify(STATE.to_status())
    except Exception as e:
        log.warning("status lookup failed: %r", e)
        return jsonify({"error": "state unavailable"}), 503


@app.get("/api/books")
def api_books():
    return jsonify(db.all_books())


@app.get("/api/books/<int:book_id>")
def api_book(book_id: int):
    book = db.get_book(book_id)
    if not book:
        abort(404)
    book["pages"] = db.book_pages(book_id)
    book["questions"] = db.book_questions(book_id)
    return jsonify(book)


@app.get("/api/books/<int:book_id>/pages")
def api_book_pages(book_id: int):
    if not db.get_book(book_id):
        abort(404)
    return jsonify(db.book_pages(book_id))


@app.get("/api/books/<int:book_id>/questions")
def api_book_questions(book_id: int):
    if not db.get_book(book_id):
        abort(404)
    return jsonify(db.book_questions(book_id))


@app.get("/api/questions/<int:qid>")
def api_question(qid: int):
    q = db.get_question(qid)
    if not q:
        abort(404)
    if q.get("book_id"):
        b = db.get_book(q["book_id"])
        if b:
            q["book_title"] = b["title"]
            q["book_author"] = b["author"]
    return jsonify(q)


@app.get("/api/vocab")
def api_vocab():
    return jsonify(db.all_vocab())


# ----- debug surface -------------------------------------------------------

@app.get("/api/debug/state")
def api_debug_state():
    try:
        from state import STATE
        return jsonify(STATE.to_debug())
    except Exception as e:
        log.warning("debug state lookup failed: %r", e)
        return jsonify({"error": "state unavailable"}), 503


@app.get("/api/debug/frame")
def api_debug_frame():
    """Latest camera frame (jpeg). Overwritten by the watch loop on each
    capture; clients should bust the cache with a cache-busting query param."""
    p = Path(FRAME_PATH)
    if not p.exists():
        abort(404)
    resp = send_file(
        str(p),
        mimetype="image/jpeg",
        max_age=0,
        last_modified=p.stat().st_mtime,
    )
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp


@app.get("/api/debug/audio")
def api_debug_audio():
    """Latest recorded question (wav). Missing until the button has been
    pressed at least once in this boot."""
    p = Path(QUESTION_WAV)
    if not p.exists():
        abort(404)
    resp = send_file(str(p), mimetype="audio/wav", max_age=0)
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp


# ----- PWA static + SPA fallback ------------------------------------------

def _static_root() -> Path:
    return Path(STATIC_DIR)


@app.get("/")
def root():
    return _serve_spa("")


@app.get("/<path:subpath>")
def catchall(subpath: str):
    # Don't hijack the API namespace
    if subpath.startswith("api/"):
        abort(404)
    return _serve_spa(subpath)


def _serve_spa(subpath: str):
    root = _static_root()
    # If we have no built bundle yet, serve a friendly placeholder
    index = root / "index.html"
    if not index.exists():
        return (
            "<html><body style='background:#0a0a0a;color:#f5a623;"
            "font-family:monospace;padding:2rem'>"
            "<h1>Lumos</h1>"
            "<p>PWA frontend not built yet. Build it with:</p>"
            "<pre>cd /home/pi/lumos/app/frontend && npm install && npm run build</pre>"
            "</body></html>",
            200,
        )
    # Try the exact path; fall through to index.html for SPA routing
    requested = root / subpath if subpath else index
    if subpath and requested.exists() and requested.is_file():
        return send_from_directory(str(root), subpath)
    return send_from_directory(str(root), "index.html")


# ----- startup helper -----------------------------------------------------

def start_in_thread() -> threading.Thread:
    def _run():
        log.info("Flask serving on %s:%d, static=%s", FLASK_HOST, FLASK_PORT, STATIC_DIR)
        app.run(
            host=FLASK_HOST,
            port=FLASK_PORT,
            debug=False,
            use_reloader=False,
            threaded=True,
        )
    t = threading.Thread(target=_run, daemon=True, name="flask")
    t.start()
    return t


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    db.init_db()
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=True, use_reloader=False)
