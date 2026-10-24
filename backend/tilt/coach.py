"""Voices (ElevenLabs) and the post-game review (Gemini). Both work without keys."""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request

from . import gemini

# ElevenLabs premade voices picked for natural, conversational delivery, one per player.
# Override any with ELEVENLABS_VOICE_<NAME>.
DEFAULT_VOICES = {
    "Narrator": "JBFqnCBsd6RMkjVDRZzb",  # George: warm British storyteller
    "Ava": "Xb7hH8MSUJpSbSDYk0k2",       # Alice: confident British woman
    "Ben": "IKne3meq5aSn9XLyUdCD",       # Charlie: laid-back Australian guy
    "Cleo": "cgSgspJ2msm6clMCkdW9",      # Jessica: young, expressive
    "Dev": "onwK4e9ZLuTAKqWW03F9",       # Daniel: calm British man
    "Eli": "pqHfZKP75CvOlQylNhV4",       # Bill: older American man
}
# Lower stability = more expressive, human-sounding delivery; style adds emotion.
VOICE_SETTINGS = {"stability": 0.35, "similarity_boost": 0.8, "style": 0.45, "use_speaker_boost": True}
TTS_MODEL = os.environ.get("ELEVENLABS_MODEL", "eleven_multilingual_v2")


def voice_for(speaker: str) -> str:
    return os.environ.get(f"ELEVENLABS_VOICE_{speaker.upper()}", DEFAULT_VOICES.get(speaker, DEFAULT_VOICES["Narrator"]))


def _post_json(url: str, body: dict, headers: dict, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


# ElevenLabs' free plan allows two requests at once; more get a 429, so queue them here.
_slots = threading.BoundedSemaphore(2)
_last_error: str | None = None


def last_error() -> str | None:
    return _last_error


def _explain(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        try:
            detail = json.loads(exc.read()).get("detail")
            msg = detail.get("message") or detail.get("status") if isinstance(detail, dict) else detail
        except Exception:
            msg = exc.reason
        return f"ElevenLabs said {exc.code}: {msg}"
    return f"Couldn't reach ElevenLabs ({exc})"


def speak(text: str, speaker: str) -> bytes | None:
    """MP3 bytes from ElevenLabs, or None (the browser then uses its own voices)."""
    global _last_error
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        _last_error = "No ELEVENLABS_API_KEY in backend/.env"
        return None
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_for(speaker)}"
    body = {"text": text, "model_id": TTS_MODEL, "voice_settings": VOICE_SETTINGS}
    with _slots:
        for attempt in range(2):
            try:
                audio = _post_json(url, body, {"xi-api-key": key, "Accept": "audio/mpeg"}, timeout=20)
                _last_error = None
                return audio
            except urllib.error.HTTPError as exc:
                _last_error = _explain(exc)
                if exc.code != 429 or attempt:
                    return None
                time.sleep(1.0)
            except Exception as exc:
                _last_error = _explain(exc)
                return None
    return None


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
    text = gemini.generate(prompt, deadline=20)
    if not text:  # the demo must not die on a flaky network
        return {"source": "template", "text": summary}
    return {"source": "gemini", "text": text.strip()}
