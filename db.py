"""SQLite persistence for Lumos.

One connection per thread is safe because Python's `sqlite3` ships with
`check_same_thread=False` + explicit locking when we pass `isolation_level=None`
and serialize writes through the single write lock SQLite already provides.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any, Iterable

from config import DB_PATH


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


SCHEMA = """
CREATE TABLE IF NOT EXISTS books (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    phash TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL DEFAULT 'Unknown',
    author TEXT NOT NULL DEFAULT 'Unknown',
    is_textbook INTEGER NOT NULL DEFAULT 0,
    current_page INTEGER NOT NULL DEFAULT 0,
    total_pages INTEGER,
    cover_path TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS pages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    book_id INTEGER NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    summary TEXT NOT NULL,
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
    is_spoiler_refusal INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_questions_book ON questions(book_id, created_at);
"""


def init_db() -> None:
    c = conn()
    with _lock:
        c.executescript(SCHEMA)
        c.commit()


def _now() -> float:
    return time.time()


# ----- books ---------------------------------------------------------------

def get_or_create_book(phash: str) -> sqlite3.Row:
    c = conn()
    with _lock:
        row = c.execute("SELECT * FROM books WHERE phash = ?", (phash,)).fetchone()
        if row is not None:
            return row
        now = _now()
        cur = c.execute(
            "INSERT INTO books (phash, created_at, updated_at) VALUES (?, ?, ?)",
            (phash, now, now),
        )
        c.commit()
        return c.execute(
            "SELECT * FROM books WHERE id = ?", (cur.lastrowid,)
        ).fetchone()


def update_book_identity(
    book_id: int, title: str, author: str, is_textbook: bool
) -> None:
    c = conn()
    with _lock:
        c.execute(
            "UPDATE books SET title=?, author=?, is_textbook=?, updated_at=? WHERE id=?",
            (title, author, 1 if is_textbook else 0, _now(), book_id),
        )
        c.commit()


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
    characters: Iterable[Any],
    vocabulary: Iterable[Any],
    concepts: Iterable[Any],
) -> int:
    c = conn()
    with _lock:
        cur = c.execute(
            """
            INSERT INTO pages
              (book_id, page_number, summary, characters_json, vocabulary_json,
               concepts_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                book_id,
                page_number,
                summary,
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
            SELECT page_number, summary, characters_json, vocabulary_json,
                   concepts_json, created_at
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
    is_spoiler_refusal: bool,
) -> int:
    c = conn()
    with _lock:
        cur = c.execute(
            """
            INSERT INTO questions
              (book_id, page_number, question, answer, is_spoiler_refusal, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                book_id,
                page_number,
                question,
                answer,
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
                        "page_number": r["page_number"],
                        "book_id": r["book_id"],
                        "book_title": r["book_title"],
                        "book_author": r["book_author"],
                        "created_at": r["created_at"],
                    }
                )
    return out


# ----- smoke test ----------------------------------------------------------

if __name__ == "__main__":
    init_db()
    b = get_or_create_book("smoketest_phash_abc123")
    print("book:", dict(b))
    update_book_identity(b["id"], "Smoke Book", "Test Author", False)
    set_current_page(b["id"], 12)
    add_page(
        b["id"],
        12,
        "On this page the detective investigates a suspicious manor.",
        [{"name": "Holmes", "role": "detective"}],
        [{"word": "perspicacious", "definition": "shrewdly discerning"}],
        ["deduction"],
    )
    add_question(b["id"], 12, "Who is Holmes?", "A detective.", False)
    print("books:", all_books())
    print("summaries:", recent_summaries(b["id"]))
    print("vocab:", all_vocab())
    print("questions:", book_questions(b["id"]))
    # Clean up so this row doesn't pollute the demo DB
    with _lock:
        conn().execute("DELETE FROM books WHERE phash='smoketest_phash_abc123'")
        conn().commit()
    print("OK")
