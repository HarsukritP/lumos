"""Gemini 2.5 Flash wrappers for book identification, page summaries,
spoiler-safe question answering, and audio transcription.

Uses the `google-genai` SDK. All calls use JSON-mode output where possible
and fall back to tolerant JSON parsing on the raw text response.
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types as gtypes

from config import GEMINI_API_KEY, MODEL_NAME

log = logging.getLogger("lumos.ai")


class AIError(RuntimeError):
    """Raised on network or upstream Gemini errors (caller shows friendly OLED)."""


class AIParseError(AIError):
    """Gemini returned content but it failed to parse as expected JSON."""


_client: genai.Client | None = None


def client() -> genai.Client:
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise AIError("GEMINI_API_KEY not set; check ~/.lumos.env")
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


# ----- OLED text caps ------------------------------------------------------
# The SSD1306 is 128x64 px. Our FONT_MD (12pt bold) fits ~14-16 chars per
# line; wrap() packs 3-5 lines. These caps are enforced server-side so a
# runaway Gemini response never breaks the OLED layout.
OLED_SUMMARY_MAX = 72       # 3-4 short lines
OLED_ANSWER_MAX = 80        # 4 short lines
OLED_TITLE_MAX = 20         # 1 line, FONT_MD — "Brothers K." style
OLED_DEFINITION_MAX = 36    # 2-3 short lines under the vocab word
OLED_ROLE_MAX = 36          # character blurb next to the name


def _clamp(text: str, limit: int) -> str:
    """Trim text to `limit` characters, ending on a word boundary where
    possible, appending a single-char ellipsis when truncated. Also strips
    markdown/newlines that would break the OLED renderer."""
    if not text:
        return ""
    s = str(text).replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\*+|_+|`+", "", s)  # markdown strip
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) <= limit:
        return s
    cut = s[: limit - 1]
    # Walk back to the last space so we don't chop a word mid-syllable.
    sp = cut.rfind(" ")
    if sp >= limit // 2:
        cut = cut[:sp]
    return cut.rstrip(",.;:—- ") + "\u2026"


# ----- JSON parsing helpers -----------------------------------------------

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _extract_json(text: str) -> Any:
    """Pull a JSON object out of a model response that may have code fences
    or stray prose. Raises AIParseError on failure."""
    if not text:
        raise AIParseError("empty response")
    cleaned = _CODE_FENCE_RE.sub("", text.strip()).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Greedy: find the outermost {...} or [...]
    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        end = cleaned.rfind(closer)
        if start != -1 and end > start:
            candidate = cleaned[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    log.warning("ai parse fallback failed; raw=%r", text[:400])
    raise AIParseError(f"non-JSON response: {text[:200]!r}")


def _image_part(path: Path | str) -> gtypes.Part:
    data = Path(path).read_bytes()
    return gtypes.Part.from_bytes(data=data, mime_type="image/jpeg")


def _call(
    prompt_parts: list[Any],
    *,
    response_json: bool = True,
    temperature: float = 0.2,
    max_retries: int = 1,
    timeout_s: float = 25.0,
) -> str:
    """Call Gemini with a list of content parts. Return raw text."""
    cfg = gtypes.GenerateContentConfig(
        temperature=temperature,
        response_mime_type="application/json" if response_json else "text/plain",
    )
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            t0 = time.monotonic()
            resp = client().models.generate_content(
                model=MODEL_NAME,
                contents=prompt_parts,
                config=cfg,
            )
            dt = time.monotonic() - t0
            text = (resp.text or "").strip()
            log.info("gemini call ok in %.2fs, %d chars", dt, len(text))
            return text
        except Exception as e:
            last_err = e
            log.warning("gemini call attempt %d failed: %r", attempt + 1, e)
            time.sleep(0.8 * (attempt + 1))
    raise AIError(f"gemini unreachable: {last_err!r}") from last_err


# ----- public functions ---------------------------------------------------

def identify_book(image_path: Path | str) -> dict:
    """Identify a physical book from one camera frame.

    Returns:
      {
        'title': str,           # full title for the app
        'author': str,          # full author for the app
        'is_textbook': bool,
        'confidence': float,    # 0..1
        'cover_visible': bool,  # True if this frame is the COVER (vs interior page)
        'oled_title': str,      # <= OLED_TITLE_MAX chars, for the tiny screen
      }
    """
    prompt = (
        "You are identifying a physical book from a single photo taken by a "
        "clip-on reading lamp's camera. The image may show the front COVER or "
        "an INTERIOR page (chapter heading, running title, or body text).\n\n"
        "Return JSON ONLY with this exact shape:\n"
        "{\n"
        '  "title": str,\n'
        '  "author": str,\n'
        '  "is_textbook": bool,\n'
        '  "confidence": float,\n'
        '  "cover_visible": bool,\n'
        f'  "oled_title": str   // <= {OLED_TITLE_MAX} chars, for a 128x64 OLED\n'
        "}\n"
        "Rules:\n"
        "- confidence is 0..1. Only return >=0.6 if you can literally read the "
        "title/author on the page; use lower scores for inferences.\n"
        '- If you cannot determine a field, use "Unknown" (never null).\n'
        '- cover_visible=true ONLY if this frame shows the front cover '
        "(big title, art). Chapter openings or text pages are cover_visible=false.\n"
        f'- oled_title must be <= {OLED_TITLE_MAX} chars. Prefer a short '
        'recognizable form (e.g. "Brothers Karamazov", "Fahrenheit 451"). '
        "No quotes, no author, no punctuation-heavy flourishes.\n"
        "No prose outside the JSON."
    )
    raw = _call([_image_part(image_path), prompt])
    data = _extract_json(raw)
    title = str(data.get("title") or "Unknown")
    return {
        "title": title,
        "author": str(data.get("author") or "Unknown"),
        "is_textbook": bool(data.get("is_textbook", False)),
        "confidence": float(data.get("confidence", 0.0) or 0.0),
        "cover_visible": bool(data.get("cover_visible", False)),
        "oled_title": _clamp(data.get("oled_title") or title, OLED_TITLE_MAX),
    }


def summarize_page(
    image_path: Path | str,
    book_title: str,
    last_known_page: int | None,
) -> dict:
    """Summarize the visible page, extract characters / vocab / concepts,
    and return Gemini's best guess at the page number.

    Shape: {
      "page_number": int,
      "summary": str (1-3 sentences),
      "characters": [{"name": str, "role": str}, ...],
      "vocabulary": [{"word": str, "definition": str}, ...],
      "concepts": [str, ...]
    }
    Vocabulary items should only include words a literate adult reader would
    find unusual or technical — this replaces the `wordfreq` rare-word filter.
    """
    prompt = (
        f"You are reading over someone's shoulder in the book \"{book_title}\". "
        f"The last page we saw was {last_known_page if last_known_page else 'the start'}. "
        "From this photo of an open book page, return JSON ONLY with this shape:\n"
        "{\n"
        '  "page_number": int,\n'
        '  "summary": "1-3 sentences, for a phone app, neutral recap of what happens on this page",\n'
        f'  "oled_summary": "<= {OLED_SUMMARY_MAX} chars, plain text, present-tense, no names if possible",\n'
        '  "characters": [{"name": str, "role": "who they are in <=10 words", '
        f'"oled_role": "<= {OLED_ROLE_MAX} chars"}}, ...],\n'
        '  "vocabulary": [{"word": str, "definition": "<=12 words for the app", '
        f'"oled_definition": "<= {OLED_DEFINITION_MAX} chars for a tiny screen"}}, ...],\n'
        '  "concepts": [str, ...]\n'
        "}\n"
        "Rules:\n"
        "- Only include vocabulary entries for words an educated adult native "
        "speaker would find unusual, archaic, technical, or foreign. Max 3 per "
        "page. Empty list is fine.\n"
        "- Characters: names that appear on the visible page. Max 4. Empty fine.\n"
        "- concepts: for textbooks, 1-3 key concept strings. For fiction, usually [].\n"
        "- page_number: read it off the page if visible; otherwise estimate from last_known_page.\n"
        f"- oled_summary MUST be <= {OLED_SUMMARY_MAX} characters, no markdown, "
        "no quotes, no trailing ellipsis (we'll add one if needed). Think of a "
        "typewritten note on an index card.\n"
        "No prose outside the JSON."
    )
    raw = _call([_image_part(image_path), prompt])
    data = _extract_json(raw)
    try:
        pn = int(data.get("page_number") or (last_known_page or 0) + 1)
    except (TypeError, ValueError):
        pn = (last_known_page or 0) + 1
    long_summary = str(data.get("summary") or "").strip()
    oled_summary = _clamp(
        data.get("oled_summary") or long_summary, OLED_SUMMARY_MAX
    )

    # Normalize characters + enforce oled_role cap.
    characters: list[dict] = []
    for c in (data.get("characters") or []):
        if not isinstance(c, dict):
            continue
        name = str(c.get("name") or "").strip()
        if not name:
            continue
        role = str(c.get("role") or "").strip()
        oled_role = _clamp(c.get("oled_role") or role, OLED_ROLE_MAX)
        characters.append({"name": name, "role": role, "oled_role": oled_role})

    # Normalize vocabulary + enforce oled_definition cap.
    vocabulary: list[dict] = []
    for v in (data.get("vocabulary") or []):
        if not isinstance(v, dict):
            continue
        word = str(v.get("word") or "").strip()
        if not word:
            continue
        definition = str(v.get("definition") or "").strip()
        oled_def = _clamp(v.get("oled_definition") or definition, OLED_DEFINITION_MAX)
        vocabulary.append({
            "word": word, "definition": definition, "oled_definition": oled_def,
        })

    return {
        "page_number": pn,
        "summary": long_summary,
        "oled_summary": oled_summary,
        "characters": characters,
        "vocabulary": vocabulary,
        "concepts": list(data.get("concepts") or []),
    }


def answer_question(
    image_path: Path | str | None,
    question: str,
    book_title: str,
    current_page: int | None,
    recent_summaries: list[dict],
) -> dict:
    """Spoiler-safe answer. Returns {'answer': str, 'refused_as_spoiler': bool}.

    The critical invariant: Gemini must never reveal content past
    `current_page`. If the user is asking about something that happens later,
    refuse in-character (friendly, not scolding).
    """
    context_lines: list[str] = []
    for s in sorted(recent_summaries, key=lambda r: r["page_number"]):
        context_lines.append(f"[p.{s['page_number']}] {s['summary']}")
    context = "\n".join(context_lines) if context_lines else "(no prior pages seen yet)"

    prompt = (
        "You are Lumos, an ambient reading companion clipped to a book.\n\n"
        f"BOOK: {book_title}\n"
        f"READER IS CURRENTLY ON: page {current_page if current_page else '?'}\n\n"
        "CONTEXT — summaries of pages the reader has already seen:\n"
        f"{context}\n\n"
        "STRICT SPOILER RULE:\n"
        f"- You may ONLY use information about the book up to and including "
        f"page {current_page if current_page else '?'}.\n"
        "- If the question is about a character, event, or idea that first "
        "appears AFTER this page, you MUST refuse by setting "
        '"refused_as_spoiler": true and replying something warm like '
        '"we haven\'t gotten there yet — ask me again once we do."\n'
        "- Do not reveal plot points, deaths, twists, or outcomes that occur "
        "past the current page, even if you're confident about them.\n"
        "- If the question is about earlier content in the book, answer helpfully.\n"
        "- If the question is unrelated to the book (weather, math), answer briefly.\n\n"
        f"QUESTION: {question}\n\n"
        "Return JSON ONLY with this shape:\n"
        "{\n"
        '  "answer": "1-3 sentences, for a phone app",\n'
        f'  "oled_answer": "<= {OLED_ANSWER_MAX} chars, for a 128x64 OLED; '
        "plain text, no markdown, no lists, no quotes; "
        'should fit the same meaning as answer in miniature",\n'
        '  "refused_as_spoiler": bool\n'
        "}\n"
        f"If refusing as a spoiler, make oled_answer <= {OLED_ANSWER_MAX} "
        'chars too (e.g. "not there yet — ask me soon"). No prose outside the JSON.'
    )
    parts: list[Any] = []
    if image_path:
        try:
            parts.append(_image_part(image_path))
        except Exception as e:
            log.warning("couldn't attach image to question: %r", e)
    parts.append(prompt)

    raw = _call(parts, temperature=0.3)
    data = _extract_json(raw)
    long_answer = str(data.get("answer") or "").strip() or "(no answer)"
    oled_answer = _clamp(data.get("oled_answer") or long_answer, OLED_ANSWER_MAX)
    return {
        "answer": long_answer,
        "oled_answer": oled_answer,
        "refused_as_spoiler": bool(data.get("refused_as_spoiler", False)),
    }


def transcribe_audio(wav_path: Path | str) -> str:
    """Transcribe a WAV recording to text using Gemini's Files API."""
    wav_path = Path(wav_path)
    if not wav_path.exists() or wav_path.stat().st_size < 1024:
        raise AIError(f"transcribe_audio: missing/empty file {wav_path}")
    try:
        uploaded = client().files.upload(file=str(wav_path))
    except Exception as e:
        raise AIError(f"file upload failed: {e!r}") from e

    prompt = (
        "Transcribe this short audio clip of a spoken question to plain text. "
        "Return ONLY the transcription, no quotes, no prose. If the audio is "
        "silent or unintelligible, return an empty string."
    )
    raw = _call([uploaded, prompt], response_json=False, temperature=0.1)
    return raw.strip().strip('"').strip()


def transcribe_and_answer(
    wav_path: Path | str,
    image_path: Path | str | None,
    book_title: str,
    current_page: int | None,
    recent_summaries: list[dict],
) -> dict:
    """Single Gemini call: transcribe the audio question AND answer it.

    Returns {
      'question': str,         # the transcribed question
      'answer': str,           # full answer for the app
      'oled_answer': str,      # short answer for OLED
      'refused_as_spoiler': bool
    }

    By combining transcription and answering into one request we eliminate
    a full Gemini round-trip (upload+transcribe then answer), cutting PTT
    latency roughly in half.
    """
    wav_path = Path(wav_path)
    if not wav_path.exists() or wav_path.stat().st_size < 1024:
        raise AIError(f"transcribe_and_answer: missing/empty file {wav_path}")
    try:
        uploaded = client().files.upload(file=str(wav_path))
    except Exception as e:
        raise AIError(f"file upload failed: {e!r}") from e

    context_lines: list[str] = []
    for s in sorted(recent_summaries, key=lambda r: r["page_number"]):
        context_lines.append(f"[p.{s['page_number']}] {s['summary']}")
    context = "\n".join(context_lines) if context_lines else "(no prior pages seen yet)"

    prompt = (
        "You are Lumos, an ambient reading companion clipped to a book.\n"
        "The attached audio is a spoken question from the reader. "
        "First transcribe the audio, then answer the question.\n\n"
        f"BOOK: {book_title}\n"
        f"READER IS CURRENTLY ON: page {current_page if current_page else '?'}\n\n"
        "CONTEXT — summaries of pages the reader has already seen:\n"
        f"{context}\n\n"
        "STRICT SPOILER RULE:\n"
        f"- You may ONLY use information about the book up to and including "
        f"page {current_page if current_page else '?'}.\n"
        "- If the question is about a character, event, or idea that first "
        "appears AFTER this page, you MUST refuse by setting "
        '"refused_as_spoiler": true and replying something warm like '
        '"we haven\'t gotten there yet — ask me again once we do."\n'
        "- Do not reveal plot points, deaths, twists, or outcomes that occur "
        "past the current page, even if you're confident about them.\n"
        "- If the question is about earlier content in the book, answer helpfully.\n"
        "- If the question is unrelated to the book (weather, math), answer briefly.\n\n"
        "Return JSON ONLY with this shape:\n"
        "{\n"
        '  "question": "the verbatim transcription of the audio",\n'
        '  "answer": "1-3 sentences, for a phone app",\n'
        f'  "oled_answer": "<= {OLED_ANSWER_MAX} chars, for a 128x64 OLED; '
        "plain text, no markdown, no lists, no quotes\",\n"
        '  "refused_as_spoiler": bool\n'
        "}\n"
        "If the audio is silent or completely unintelligible, set "
        '"question": "" and "answer": "". No prose outside the JSON.'
    )
    parts: list[Any] = [uploaded]
    if image_path:
        try:
            parts.append(_image_part(image_path))
        except Exception as e:
            log.warning("couldn't attach image to question: %r", e)
    parts.append(prompt)

    raw = _call(parts, temperature=0.3)
    data = _extract_json(raw)
    question = str(data.get("question") or "").strip()
    long_answer = str(data.get("answer") or "").strip()
    oled_answer = _clamp(data.get("oled_answer") or long_answer, OLED_ANSWER_MAX)
    return {
        "question": question,
        "answer": long_answer or "(no answer)",
        "oled_answer": oled_answer,
        "refused_as_spoiler": bool(data.get("refused_as_spoiler", False)),
    }


# ----- smoke test ----------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 ai.py <identify|summarize|ask|transcribe> [args...]")
        print("  identify <image.jpg>")
        print("  summarize <image.jpg> <title> <last_page>")
        print("  ask <question>")
        print("  transcribe <audio.wav>")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    mode = sys.argv[1]
    if mode == "identify":
        print(json.dumps(identify_book(sys.argv[2]), indent=2))
    elif mode == "summarize":
        title = sys.argv[3] if len(sys.argv) > 3 else "Unknown"
        last_page = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        print(json.dumps(summarize_page(sys.argv[2], title, last_page), indent=2))
    elif mode == "ask":
        out = answer_question(
            None,
            " ".join(sys.argv[2:]),
            "Test",
            1,
            [{"page_number": 1, "summary": "introduction; nothing has happened yet."}],
        )
        print(json.dumps(out, indent=2))
    elif mode == "transcribe":
        print(repr(transcribe_audio(sys.argv[2])))
    else:
        print(f"unknown mode {mode!r}")
        sys.exit(2)
