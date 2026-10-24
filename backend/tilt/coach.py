"""Voices (ElevenLabs) and the post-game review (Gemini). Both work without keys."""
from __future__ import annotations

import json
import os
import urllib.request

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

from .voices import DEFAULT_VOICES, voice_for  # noqa: E402,F401  (kept for existing imports)


def _post_json(url: str, body: dict, headers: dict, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def speak(text: str, speaker: str) -> bytes | None:
    """MP3 bytes from ElevenLabs (see voices.py), or None (the browser then uses its own voices)."""
    from .voices import speak as eleven
    return eleven(text, speaker)


def review(report: dict, summary: str) -> dict:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return {"source": "template", "text": summary}
    slim = {k: v for k, v in report.items() if k != "timeline"}
    slim["sample_answers"] = [{"said": t["text"], "lie": t["lie"], "read": t["p_lie"], "cues": t["cues"]}
                              for t in report.get("timeline", [])[-8:]]
    prompt = (
        "You are a blunt deception coach reviewing a spoken Mafia session. A model read the player's face "
        "(blinks, gaze, brows, lips), voice (pitch, speed, pauses) and words to spot lies. Using only the JSON "
        "below, write 4 short sentences for the player, spoken aloud: how readable their lies were, the tells "
        "that gave them away, what it cost them in the game, and one drill to fix it. No markdown.\n\n"
        + json.dumps(slim)
    )
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    try:
        raw = _post_json(url, {"contents": [{"parts": [{"text": prompt}]}]}, {"x-goog-api-key": key})
        return {"source": "gemini", "text": json.loads(raw)["candidates"][0]["content"]["parts"][0]["text"].strip()}
    except Exception as exc:  # the demo must not die on a flaky network
        return {"source": "template", "text": summary, "error": str(exc)}
