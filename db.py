"""SQLite persistence for Lumos.

One connection, shared across threads with `check_same_thread=False`, all
writes serialized through the single module-level `_lock`. WAL mode lets the
Flask read-threads run concurrently with the orchestrator's writes.

Dedup rule: books are keyed by `(title_key, author_key)` — the lowercased,
whitespace-normalized title and author — NOT by image phash. Cover photos
jitter enough that phash-based keys create ghost "books" for the same
physical book. Phash is still stored as `cover_phash` for diagnostics and
as a weak same-cover hint, just not as the identity.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from config import DB_PATH

log = logging.getLogger("lumos.db")

_lock = threading.Lock()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


_conn: sqlite3.Connection | None = None


def conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = _connect()
    return _conn


SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS books (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title_key TEXT NOT NULL,
    author_key TEXT NOT NULL,
    title TEXT NOT NULL,
    author TEXT NOT NULL,
    is_textbook INTEGER NOT NULL DEFAULT 0,
    current_page INTEGER NOT NULL DEFAULT 0,
    total_pages INTEGER,
    cover_phash TEXT,
    cover_path TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(title_key, author_key)
);

CREATE TABLE IF NOT EXISTS pages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    book_id INTEGER NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    summary TEXT NOT NULL,
    oled_summary TEXT NOT NULL DEFAULT '',
    characters_json TEXT NOT NULL DEFAULT '[]',
    vocabulary_json TEXT NOT NULL DEFAULT '[]',
    concepts_json TEXT NOT NULL DEFAULT '[]',
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pages_book ON pages(book_id, page_number);

CREATE TABLE IF NOT EXISTS questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    book_id INTEGER REFERENCES books(id) ON DELETE CASCADE,
    page_number INTEGER,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    oled_answer TEXT NOT NULL DEFAULT '',
    is_spoiler_refusal INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_questions_book ON questions(book_id, created_at);
"""


def _current_schema_version() -> int:
    c = conn()
    try:
        row = c.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    except sqlite3.OperationalError:
        return 0
    if not row:
        return 0
    try:
        return int(row["value"])
    except (TypeError, ValueError):
        return 0


def init_db() -> None:
    """Create the schema; if an older schema is on disk, drop and recreate.

    We're pre-launch; no migration data is worth preserving. If a user has
    bound data they care about, they would have asked for a migration path."""
    c = conn()
    with _lock:
        existing = _current_schema_version()
        if existing and existing < SCHEMA_VERSION:
            log.warning(
                "db schema v%d < v%d; wiping and recreating",
                existing, SCHEMA_VERSION,
            )
            _drop_all_tables(c)
        elif existing == 0:
            # Could be either a blank DB or a v1 schema with no meta row.
            # If books table already exists without title_key, it's v1 → drop.
            cols = [r["name"] for r in c.execute("PRAGMA table_info(books)")]
            if cols and "title_key" not in cols:
                log.warning("legacy books schema detected; wiping")
                _drop_all_tables(c)
        c.executescript(SCHEMA)
        c.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        c.commit()


def _drop_all_tables(c: sqlite3.Connection) -> None:
    # Order matters for FK constraints, but since we're dropping everything
    # in one transaction and PRAGMA foreign_keys can be momentarily turned
    # off, just execute unconditionally.
    c.execute("PRAGMA foreign_keys=OFF")
    for name in ("questions", "pages", "books", "meta"):
        c.execute(f"DROP TABLE IF EXISTS {name}")
    c.execute("PRAGMA foreign_keys=ON")


def reset() -> dict:
    """Wipe all user-visible rows (books/pages/questions). Schema stays.
    Returns before/after row counts so callers can show a summary."""
    before = stats()
    c = conn()
    with _lock:
        c.execute("PRAGMA foreign_keys=OFF")
        c.execute("DELETE FROM questions")
        c.execute("DELETE FROM pages")
        c.execute("DELETE FROM books")
        # Reset AUTOINCREMENT counters so IDs restart at 1 after a reset.
        try:
            c.execute("DELETE FROM sqlite_sequence")
        except sqlite3.OperationalError:
            pass
        c.execute("PRAGMA foreign_keys=ON")
        c.commit()
    after = stats()
    log.info("db reset: %s -> %s", before["counts"], after["counts"])
    return {"before": before, "after": after}


def stats() -> dict:
    """Row counts + on-disk footprint. Safe to call from any thread."""
    c = conn()
    counts: dict[str, int] = {}
    with _lock:
        for table in ("books", "pages", "questions"):
            try:
                n = c.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
            except sqlite3.OperationalError:
                n = 0
            counts[table] = int(n)
    size = 0
    paths = []
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(DB_PATH) + suffix)
        if p.exists():
            size += p.stat().st_size
            paths.append(str(p))
    return {
        "db_path": str(DB_PATH),
        "schema_version": SCHEMA_VERSION,
        "size_bytes": size,
        "files": paths,
        "counts": counts,
    }


def _now() -> float:
    return time.time()


# ----- identity normalization ---------------------------------------------

_WS_RE = re.compile(r"\s+")


def normalize_identity(title: str, author: str) -> tuple[str, str]:
    """Canonical form used for dedup. Lowercased, punctuation-stripped, and
    whitespace-collapsed. Must be stable across Gemini responses for the
    same physical book.

    Examples:
      ("The Brothers Karamazov", "Fyodor Dostoevsky")
        -> ("the brothers karamazov", "fyodor dostoevsky")
      ("  The  Brothers  Karamazov!", "Dostoevsky, Fyodor")
        -> ("the brothers karamazov", "dostoevsky fyodor")
    """
    def norm(s: str) -> str:
        s = (s or "").strip().lower()
        # strip most punctuation; keep letters, digits, spaces, hyphen
        s = re.sub(r"[^\w\-\s]", " ", s, flags=re.UNICODE)
        s = _WS_RE.sub(" ", s).strip()
        return s
    return norm(title), norm(author)


# ----- books ---------------------------------------------------------------

def find_book_by_identity(title: str, author: str) -> dict | None:
    tk, ak = normalize_identity(title, author)
    if not tk:
        return None
    c = conn()
    with _lock:
        row = c.execute(
            "SELECT * FROM books WHERE title_key=? AND author_key=?", (tk, ak)
        ).fetchone()
    return dict(row) if row else None


def find_or_create_book_by_identity(
    title: str,
    author: str,
    is_textbook: bool,
    cover_phash: str | None = None,
) -> dict:
    """Idempotent: same (title, author) returns the same row across calls.
    Creates it (or updates identity fields) on first sight."""
    tk, ak = normalize_identity(title, author)
    if not tk:
        raise ValueError("cannot create book with empty title")
    c = conn()
    now = _now()
    with _lock:
        row = c.execute(
            "SELECT * FROM books WHERE title_key=? AND author_key=?", (tk, ak)
        ).fetchone()
        if row is not None:
            # Refresh descriptive fields if they improved.
            c.execute(
                "UPDATE books SET title=?, author=?, is_textbook=?, "
                "cover_phash=COALESCE(?, cover_phash), updated_at=? WHERE id=?",
                (title, author, 1 if is_textbook else 0,
                 cover_phash, now, row["id"]),
            )
            c.commit()
            row = c.execute(
                "SELECT * FROM books WHERE id=?", (row["id"],)
            ).fetchone()
            return dict(row)
        cur = c.execute(
            "INSERT INTO books (title_key, author_key, title, author, "
            "is_textbook, cover_phash, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (tk, ak, title, author, 1 if is_textbook else 0, cover_phash, now, now),
        )
        c.commit()
        row = c.execute(
            "SELECT * FROM books WHERE id=?", (cur.lastrowid,)
        ).fetchone()
    return dict(row)


def set_current_page(book_id: int, page_number: int) -> None:
    c = conn()
    with _lock:
        c.execute(
            "UPDATE books SET current_page=?, updated_at=? WHERE id=?",
            (page_number, _now(), book_id),
        )
        c.commit()


def all_books() -> list[dict]:
    c = conn()
    with _lock:
        rows = c.execute(
            """
            SELECT b.*,
              (SELECT COUNT(*) FROM pages WHERE book_id=b.id) AS page_count,
              (SELECT COUNT(*) FROM questions WHERE book_id=b.id) AS question_count
            FROM books b
            ORDER BY b.updated_at DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]


def get_book(book_id: int) -> dict | None:
    c = conn()
    with _lock:
        row = c.execute("SELECT * FROM books WHERE id=?", (book_id,)).fetchone()
        return dict(row) if row else None


# ----- pages ---------------------------------------------------------------

def add_page(
    book_id: int,
    page_number: int,
    summary: str,
    oled_summary: str,
    characters: Iterable[Any],
    vocabulary: Iterable[Any],
    concepts: Iterable[Any],
) -> int:
    c = conn()
    with _lock:
        cur = c.execute(
            """
            INSERT INTO pages
              (book_id, page_number, summary, oled_summary, characters_json,
               vocabulary_json, concepts_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                book_id,
                page_number,
                summary,
                oled_summary,
                json.dumps(list(characters)),
                json.dumps(list(vocabulary)),
                json.dumps(list(concepts)),
                _now(),
            ),
        )
        c.commit()
        return cur.lastrowid


def recent_summaries(book_id: int, n: int = 5) -> list[dict]:
    c = conn()
    with _lock:
        rows = c.execute(
            """
            SELECT page_number, summary, oled_summary, characters_json,
                   vocabulary_json, concepts_json, created_at
            FROM pages
            WHERE book_id=?
            ORDER BY page_number DESC
            LIMIT ?
            """,
            (book_id, n),
        ).fetchall()
    out: list[dict] = []
    for r in rows:
        out.append(
            {
                "page_number": r["page_number"],
                "summary": r["summary"],
                "oled_summary": r["oled_summary"],
                "characters": json.loads(r["characters_json"]),
                "vocabulary": json.loads(r["vocabulary_json"]),
                "concepts": json.loads(r["concepts_json"]),
                "created_at": r["created_at"],
            }
        )
    return out


def book_pages(book_id: int) -> list[dict]:
    c = conn()
    with _lock:
        rows = c.execute(
            """
            SELECT * FROM pages WHERE book_id=? ORDER BY page_number ASC
            """,
            (book_id,),
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["characters"] = json.loads(d.pop("characters_json"))
        d["vocabulary"] = json.loads(d.pop("vocabulary_json"))
        d["concepts"] = json.loads(d.pop("concepts_json"))
        out.append(d)
    return out


# ----- questions -----------------------------------------------------------

def add_question(
    book_id: int | None,
    page_number: int | None,
    question: str,
    answer: str,
    oled_answer: str,
    is_spoiler_refusal: bool,
) -> int:
    c = conn()
    with _lock:
        cur = c.execute(
            """
            INSERT INTO questions
              (book_id, page_number, question, answer, oled_answer,
               is_spoiler_refusal, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                book_id,
                page_number,
                question,
                answer,
                oled_answer,
                1 if is_spoiler_refusal else 0,
                _now(),
            ),
        )
        c.commit()
        return cur.lastrowid


def book_questions(book_id: int) -> list[dict]:
    c = conn()
    with _lock:
        rows = c.execute(
            "SELECT * FROM questions WHERE book_id=? ORDER BY created_at DESC",
            (book_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_question(qid: int) -> dict | None:
    c = conn()
    with _lock:
        row = c.execute("SELECT * FROM questions WHERE id=?", (qid,)).fetchone()
        return dict(row) if row else None


# ----- vocab ---------------------------------------------------------------

def all_vocab() -> list[dict]:
    """Return every vocabulary entry joined with its book, newest first."""
    c = conn()
    with _lock:
        rows = c.execute(
            """
            SELECT p.vocabulary_json, p.page_number, p.created_at,
                   b.id AS book_id, b.title AS book_title, b.author AS book_author
            FROM pages p
            JOIN books b ON b.id = p.book_id
            ORDER BY p.created_at DESC
            """
        ).fetchall()
    out: list[dict] = []
    for r in rows:
        vocab = json.loads(r["vocabulary_json"])
        for v in vocab:
            if isinstance(v, dict) and v.get("word"):
                out.append(
                    {
                        "word": v.get("word", ""),
                        "definition": v.get("definition", ""),
                        "oled_definition": v.get("oled_definition", ""),
                        "page_number": r["page_number"],
                        "book_id": r["book_id"],
                        "book_title": r["book_title"],
                        "book_author": r["book_author"],
                        "created_at": r["created_at"],
                    }
                )
    return out


# ----- inspector (raw table dump for admin view) ---------------------------

INSPECTOR_TABLES = ("meta", "books", "pages", "questions")


def dump_tables(limit_per_table: int = 200) -> dict:
    """Raw JSON-able dump of each user-visible table for the admin view.
    Truncates JSON blobs so a bloated row doesn't break the UI."""
    c = conn()
    out: dict[str, Any] = {}
    with _lock:
        for t in INSPECTOR_TABLES:
            try:
                cols = [r["name"] for r in c.execute(f"PRAGMA table_info({t})")]
                rows = c.execute(
                    f"SELECT * FROM {t} LIMIT ?", (limit_per_table,)
                ).fetchall()
                out[t] = {
                    "columns": cols,
                    "rows": [dict(r) for r in rows],
                    "count": c.execute(f"SELECT COUNT(*) AS n FROM {t}").fetchone()["n"],
                }
            except sqlite3.OperationalError as e:
                out[t] = {"error": str(e)}
    return out


# ----- smoke test ----------------------------------------------------------

if __name__ == "__main__":
    # IMPORTANT: run the smoke against a throwaway DB file so a crash mid-
    # test can't leave fake books in ~/lumos.db. Previously this used the
    # real DB and relied on a cleanup DELETE that was skipped on failure.
    import os
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="lumos_db_smoke_"))
    os.environ["LUMOS_DB"] = str(tmp / "smoke.db")
    # Re-point the module-level DB_PATH for this process.
    import config
    config.DB_PATH = Path(os.environ["LUMOS_DB"])
    globals()["DB_PATH"] = config.DB_PATH
    # Drop any cached connection from prior imports.
    _conn = None

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    init_db()
    print("db:", config.DB_PATH)
    print("stats:", stats())

    b = find_or_create_book_by_identity(
        "The Brothers Karamazov", "Fyodor Dostoevsky", is_textbook=False,
        cover_phash="smoke_abc123",
    )
    print("book:", b)
    # Dedup is case-insensitive + punctuation/whitespace tolerant, but it
    # does NOT reorder tokens — ("Last, First" vs "First Last") counts as
    # two different authors. Gemini returns a consistent form in practice.
    b2 = find_or_create_book_by_identity(
        "  the  Brothers Karamazov!", "Fyodor DOSTOEVSKY", is_textbook=False,
    )
    assert b2["id"] == b["id"], f"dedup broken: {b['id']} vs {b2['id']}"
    print("dedup ok (case + punctuation + whitespace tolerant)")
    add_page(
        b["id"], 312,
        "Ivan's anxiety sharpens as the evening cools over the courtyard.",
        "Ivan tense in courtyard.",
        [{"name": "Ivan", "role": "brother"}],
        [{"word": "perspicacious", "definition": "shrewdly discerning", "oled_definition": "shrewd"}],
        [],
    )
    add_question(b["id"], 312, "Who is Ivan?", "Ivan is the middle Karamazov brother, a rationalist.",
                 "Ivan: middle brother, rationalist.", False)
    print("recent:", recent_summaries(b["id"]))
    print("vocab:", all_vocab())

    # Throwaway DB; just nuke the whole directory.
    import shutil
    if _conn is not None:
        _conn.close()
    shutil.rmtree(tmp, ignore_errors=True)
    print("OK (smoke DB removed)")
