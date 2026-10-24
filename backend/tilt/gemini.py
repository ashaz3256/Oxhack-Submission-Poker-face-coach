"""Calling Gemini, with fallbacks: older model names get retired and busy models return 503, and
the game must never wait long or die because of it. We try a short list of models and stick with
whichever answered last."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
# Lite first: it is fast enough for a live call. The -latest aliases track Google's newest models.
DEFAULT_MODELS = ["gemini-flash-lite-latest", "gemini-flash-latest", "gemini-2.5-flash-lite"]

_retired: set[str] = set()
_preferred: str | None = None


def models() -> list[str]:
    chosen = os.environ.get("GEMINI_MODEL")
    order = ([_preferred] if _preferred else []) + ([chosen] if chosen else []) + DEFAULT_MODELS
    seen = []
    for m in order:
        if m and m not in seen and m not in _retired:
            seen.append(m)
    return seen


def generate(prompt: str, *, json_mode: bool = False, temperature: float | None = None,
             deadline: float = 10.0) -> str | None:
    """The model's text, or None if there is no key or nothing answered within `deadline` seconds."""
    global _preferred
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    config = {}
    if json_mode:
        config["responseMimeType"] = "application/json"
    if temperature is not None:
        config["temperature"] = temperature
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}], "generationConfig": config}).encode()
    stop = time.time() + deadline
    for model in models():
        left = stop - time.time()
        if left < 1:
            break
        req = urllib.request.Request(URL.format(model=model), data=body,
                                     headers={"Content-Type": "application/json", "x-goog-api-key": key})
        try:
            with urllib.request.urlopen(req, timeout=left) as resp:
                raw = json.loads(resp.read())
            parts = raw["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
            _preferred = model
            return text
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                _retired.add(model)   # this model name is gone for this key
            if exc.code in (401, 403):
                return None           # bad key: no point trying other models
        except Exception:
            pass                      # timeout, 503 or a malformed reply: try the next model
    return None


def generate_json(prompt: str, **kw):
    text = generate(prompt, json_mode=True, **kw)
    if text is None:
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None
