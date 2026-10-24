"""Live coaching lines, the Gemini session review and ElevenLabs speech. Every call works without keys."""
from __future__ import annotations

import json
import os
import urllib.request

from .features import TELL_FEATURES, vector, vitals_features
from .game import Session
from .report import headline, leak_report

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
ELEVEN_VOICE = os.environ.get("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")


def live_read(session: Session) -> dict:
    """What the Shark sees right now, and what the coach says about it."""
    now = session.clock()
    f = vitals_features(session.samples, now)
    hand = session.current_hand
    f["decision_s"] = now - hand.street_started_at if hand and hand.phase != "done" else 0.0
    p_bluff = session.tell_model.predict(vector(f, TELL_FEATURES))
    latest = session.samples[-1] if session.samples else None
    hr_z = f["hr_z"]
    if not session.tell_model.ready:
        line = "The Shark is still studying you."
    elif p_bluff > 0.6:
        line = f"You're broadcasting. If you bet now, the Shark reads {p_bluff:.0%} bluff."
    elif hr_z > 1.5:
        line = "Heart rate is up. Slow your breathing before you act."
    else:
        line = "Poker face holding."
    return {
        "hr": None if latest is None else latest.hr,
        "br": None if latest is None else latest.br,
        "hr_z": round(hr_z, 2),
        "shark_p_bluff": round(p_bluff, 3),
        "shark_ready": session.tell_model.ready,
        "line": line,
    }


def _post_json(url: str, body: dict, headers: dict) -> bytes:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def session_review(session: Session) -> dict:
    report = leak_report(session)
    report["tells"].pop("curve", None)
    return review_from_report(report, headline(report), game="poker")


def review_from_report(report: dict, summary: str, game: str) -> dict:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return {"source": "template", "text": summary}
    prompt = (
        f"You are a blunt coach reviewing a {game} session where an AI read the player's heart rate and breathing "
        "to tell when they were bluffing or lying. Using only the JSON below, write 4 short sentences: how much the "
        "player's body gave away, which signal gave them away, what it cost them, and one concrete drill for next "
        "session. No markdown.\n\n" + json.dumps(report)
    )
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    try:
        raw = _post_json(url, {"contents": [{"parts": [{"text": prompt}]}]}, {"x-goog-api-key": key})
        text = json.loads(raw)["candidates"][0]["content"]["parts"][0]["text"]
        return {"source": "gemini", "text": text.strip()}
    except Exception as exc:  # the demo must not die on a flaky network
        return {"source": "template", "text": summary, "error": str(exc)}


def speak(text: str) -> bytes | None:
    """MP3 bytes from ElevenLabs, or None when no key is set (the browser falls back to speechSynthesis)."""
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        return None
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE}"
    return _post_json(url, {"text": text, "model_id": "eleven_flash_v2_5"}, {"xi-api-key": key, "Accept": "audio/mpeg"})
