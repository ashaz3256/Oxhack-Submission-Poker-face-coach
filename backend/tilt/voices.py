"""Expressive delivery on top of coach.py's ElevenLabs voices: audio tags, per-character tweaks, a cache.

coach.py chooses each player's voice and the base settings. This module tries the expressive
real-time model first (`eleven_v4_turbo`, which understands delivery tags like [whispers]); if that
fails on this account it falls back to coach.py's model with the tags stripped, remembers that, and
carries on. So the worst case is exactly the previous behaviour.

    ELEVENLABS_EXPRESSIVE_MODEL=eleven_v4_turbo   # set to "off" to skip it entirely
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections import OrderedDict

API = "https://api.elevenlabs.io/v1/text-to-speech/{voice}?output_format=mp3_44100_128"
TAG_MODELS = {"eleven_v4_turbo", "eleven_v4", "eleven_v3", "eleven_v3_conversational"}

# Nudges on top of coach.VOICE_SETTINGS. Lower stability = more expressive; style exaggerates character.
TWEAKS = {
    "Narrator": {"stability": 0.45, "style": 0.35, "speed": 0.95},  # steady host
    "Ava": {"stability": 0.45, "speed": 1.03},                      # crisp
    "Ben": {"style": 0.5},                                         # loose, jokey
    "Cleo": {"stability": 0.25, "style": 0.55, "speed": 1.08},      # anxious
    "Dev": {"stability": 0.6, "style": 0.2},                        # calm
    "Eli": {"stability": 0.3, "style": 0.6, "speed": 0.93},         # theatrical
}

# Documented ElevenLabs delivery tags we allow; anything else in [brackets] is removed.
TAGS = ["whispers", "sighs", "laughs", "sarcastic", "curious", "excited", "mischievously", "shouting",
        "nervous", "exhales"]
TAG_RE = re.compile(r"\[([a-zA-Z ]{2,24})\]\s*")

# Narrator lines that are always delivered a certain way.
DIRECTIONS = [
    (re.compile(r"^(Night falls|Mafia, open your eyes|Who\? Whisper)"), "whispers"),
    (re.compile(r"found dead"), "sighs"),
    (re.compile(r"You win!"), "excited"),
    (re.compile(r"You lose\."), "mischievously"),
]

_cache: "OrderedDict[tuple, bytes]" = OrderedDict()
CACHE_SIZE = 300
_broken_models: set[str] = set()


def strip_tags(text: str) -> str:
    return TAG_RE.sub("", text).strip()


def direct(speaker: str, text: str) -> str:
    """Keep allowed tags, drop unknown ones, and add the narrator's fixed directions."""
    text = TAG_RE.sub(lambda m: f"[{m.group(1).lower()}] " if m.group(1).lower() in TAGS else "", text).strip()
    if speaker == "Narrator" and not TAG_RE.match(text):
        for pattern, tag in DIRECTIONS:
            if pattern.search(text):
                return f"[{tag}] {text}"
    return text


def _post(voice: str, body: dict, key: str) -> bytes:
    req = urllib.request.Request(API.format(voice=voice), data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "xi-api-key": key, "Accept": "audio/mpeg"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def synthesize(text: str, speaker: str, voice: str, base_settings: dict, fallback_model: str,
               post=None) -> bytes | None:
    """MP3 bytes, or None so the browser uses its own voice. Raises the last error if every model fails.

    `post(voice, body, key)` sends one request; coach.py passes its own, which queues and retries.
    """
    post = post or _post
    key = os.environ.get("ELEVENLABS_API_KEY")
    text = (text or "").strip()
    if not key or not text:
        return None
    expressive = os.environ.get("ELEVENLABS_EXPRESSIVE_MODEL", "eleven_v4_turbo")
    settings = {**base_settings, **TWEAKS.get(speaker, {})}
    attempts = []
    if expressive != "off" and expressive not in _broken_models and expressive != fallback_model:
        attempts.append((expressive, expressive in TAG_MODELS))
    attempts.append((fallback_model, fallback_model in TAG_MODELS))
    error: Exception | None = None
    for model, tags in attempts:
        spoken = direct(speaker, text) if tags else strip_tags(text)
        cache_key = (voice, model, spoken)
        if cache_key in _cache:
            _cache.move_to_end(cache_key)
            return _cache[cache_key]
        try:
            audio = post(voice, {"text": spoken, "model_id": model, "voice_settings": settings}, key)
        except urllib.error.HTTPError as exc:
            if exc.code in (400, 403, 422) and model != fallback_model:
                _broken_models.add(model)  # not available on this account; stop trying it
            error = exc
            continue
        except Exception as exc:
            error = exc
            continue
        _cache[cache_key] = audio
        if len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)
        return audio
    if error is not None:
        raise error
    return None
