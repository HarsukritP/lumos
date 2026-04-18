#!/usr/bin/env python3
"""Render every OLED screen variant as a 4x-scaled PNG for visual QA.

Run:  python3 simulate_display.py
Output: /tmp/lumos/sim_*.png  (one per screen)
"""
from pathlib import Path
import display

OUT = Path("/tmp/lumos/sim")
OUT.mkdir(parents=True, exist_ok=True)


def save(name: str) -> None:
    png = display.snapshot_png()
    p = OUT / f"{name}.png"
    p.write_bytes(png)
    print(f"  {p}  ({len(png)} bytes)")


print("=== Lumos OLED simulation ===\n")

# 1. Welcome QR
display.show_qr("http://lumos.local:8080/library")
save("01_qr_welcome")

# 2. Hunting
display.show_status("Lumos", ["looking for", "a book..."])
save("02_hunting")

# 3. Identifying (progress)
display.show_status("Lumos", ['saw "Brothers K."', "1/2 confirmed"])
save("03_identifying")

# 4. Resume prompt
display.show_resume_prompt("Brothers Karamazov", 312)
save("04_resume_prompt")

# 5. Reading page
display.show_status("Lumos", ["reading", "the page..."])
save("05_reading_page")

# 6. Page summary
display.show_page_summary(
    47, "Ivan challenges Alyosha at the tavern, building toward the Grand Inquisitor."
)
save("06_page_summary")

# 7. Caught up (main idle display)
display.show_caught_up(47, "Brothers Karamazov")
save("07_caught_up")

# 8. Vocab card
display.show_vocab("PERSPICACIOUS", "shrewdly discerning; notably perceptive")
save("08_vocab_card")

# 9. Character card
display.show_character("Ivan Karamazov", "eldest legitimate son; philosopher, atheist")
save("09_character_card")

# 10. PTT listening (footer overlay on caught-up)
display.show_ptt_footer(
    {"type": "caught_up", "page": 47, "title": "Brothers Karamazov"},
    "\u25cf listening\u2026",
)
save("10_ptt_listening")

# 11. PTT recording (footer overlay with timer)
display.show_ptt_footer(
    {"type": "caught_up", "page": 47, "title": "Brothers Karamazov"},
    "\u25cf rec 02.4s",
)
save("11_ptt_recording")

# 12. PTT recording over vocab card
display.show_ptt_footer(
    {"type": "card", "card": {"type": "vocab", "word": "PERSPICACIOUS",
     "definition": "shrewdly discerning"}},
    "\u25cf rec 01.8s",
)
save("12_ptt_over_vocab")

# 13. Thinking
display.show_status("Lumos", ["thinking\u2026"])
save("13_thinking")

# 14. Answer
display.show_answer(
    "Smerdyakov is a servant at the Karamazov estate introduced on p. 94."
)
save("14_answer")

# 15. Spoiler refusal
display.show_answer("not there yet \u2014 ask me soon", refused=True)
save("15_spoiler_refused")

# 16. Short press hint
display.show_status("Lumos", ["press &", "hold to talk"])
save("16_press_hold")

# 17. Ready / boot
display.show_status("Lumos", ["booting..."])
save("17_booting")

# 18. No signal
display.show_status("Lumos", ["no signal,", "try again"])
save("18_no_signal")

print(f"\nDone: {len(list(OUT.glob('*.png')))} screens rendered to {OUT}")
