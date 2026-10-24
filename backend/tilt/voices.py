"""Character voices on ElevenLabs: per-player delivery, audio tags, model fallback and a cache.

Default model is `eleven_v4_turbo` (real-time, supports audio tags like [whispers]). If a request
fails (plan, quota, model not available) we drop to `eleven_flash_v2_5` with the tags stripped,
remember that, and carry on. With no key at all the browser voices take over, as before.

Configure in backend/.env:
    ELEVENLABS_API_KEY=...
    ELEVENLABS_MODEL=eleven_v4_turbo          # optional
    ELEVENLABS_VOICE_NARRATOR=<voice id>      # optional, per player: _AVA _BEN _CLEO _DEV _ELI
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
FALLBACK_MODEL = "eleven_flash_v2_5"

# ElevenLabs premade voices; swap any for a Voice Library character with ELEVENLABS_VOICE_<NAME>.
DEFAULT_VOICES = {
    "Narrator": "JBFqnCBsd6RMkjVDRZzb",  # George
    "Ava": "EXAVITQu4vr4xnSDxMaL",       # Sarah
    "Ben": "pNInz6obpgDQGcFmaJgB",       # Adam
    "Cleo": "21m00Tcm4TlvDq8ikWAM",      # Rachel
    "Dev": "ErXwobaYiN019PkySvjV",       # Antoni
    "Eli": "VR6AewLTigWG4xSOukaG",       # Arnold
}

# Lower stability = more expressive and variable; style exaggerates the voice's character.
PROFILES = {
    "Narrator": {"stability": 0.45, "similarity_boost": 0.8, "style": 0.35, "speed": 0.95},  # noir host
    "Ava": {"stability": 0.55, "similarity_boost": 0.8, "style": 0.3, "speed": 1.05},        # ex-lawyer, crisp
    "Ben": {"stability": 0.35, "similarity_boost": 0.75, "style": 0.45, "speed": 1.05},      # joker, loose
    "Cleo": {"stability": 0.25, "similarity_boost": 0.75, "style": 0.55, "speed": 1.12},     # anxious, fast
    "Dev": {"stability": 0.7, "similarity_boost": 0.8, "style": 0.15, "speed": 0.98},        # calm analyst
    "Eli": {"stability": 0.3, "similarity_boost": 0.8, "style": 0.6, "speed": 0.9},          # theatrical
}

# Tags we allow (documented ElevenLabs delivery tags). Anything else in [brackets] is removed.
TAGS = ["whispers", "sighs", "laughs", "sarcastic", "curious", "excited", "mischievously", "shouting",
        "nervous", "exhales"]
TAG_RE = re.compile(r"\[([a-zA-Z ]{2,24})\]\s*")

# Lines the narrator always delivers a certain way.
DIRECTIONS = [
    (re.compile(r"^(Night falls|Mafia, open your eyes|Who\? Whisper)"), "whispers"),
    (re.compile(r"found dead"), "sighs"),
    (re.compile(r"You win!"), "excited"),
    (re.compile(r"You lose\."), "mischievously"),
]

_cache: "OrderedDict[tuple, bytes]" = OrderedDict()
CACHE_SIZE = 300
_broken_models: set[str] = set()


def voice_for(speaker: str) -> str:
    return os.environ.get(f"ELEVENLABS_VOICE_{speaker.upper()}", DEFAULT_VOICES.get(speaker, DEFAULT_VOICES["Narrator"]))


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


def speak(text: str, speaker: str) -> bytes | None:
    """MP3 bytes, or None so the browser uses its own voice."""
    key = os.environ.get("ELEVENLABS_API_KEY")
    text = (text or "").strip()
    if not key or not text:
        return None
    model = os.environ.get("ELEVENLABS_MODEL", "eleven_v4_turbo")
    attempts = []
    if model not in _broken_models:
        attempts.append((model, model in TAG_MODELS, True))
    attempts.append((FALLBACK_MODEL, False, False))
    voice = voice_for(speaker)
    for m, tags, settings in attempts:
        spoken = direct(speaker, text) if tags else strip_tags(text)
        cache_key = (voice, m, spoken)
        if cache_key in _cache:
            _cache.move_to_end(cache_key)
            return _cache[cache_key]
        body = {"text": spoken, "model_id": m}
        if settings:
            body["voice_settings"] = PROFILES.get(speaker, PROFILES["Narrator"])
        try:
            audio = _post(voice, body, key)
        except urllib.error.HTTPError as exc:
            if exc.code in (400, 403, 422) and m != FALLBACK_MODEL:
                _broken_models.add(m)  # this model or its settings are not available on this account
            continue
        except Exception:
            continue
        _cache[cache_key] = audio
        if len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)
        return audio
    return None
