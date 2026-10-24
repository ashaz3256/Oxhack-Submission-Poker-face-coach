"""Generate the game's sound effects once with ElevenLabs and save them as files.

    cd backend
    python scripts/make_sfx.py            # skips files that already exist
    python scripts/make_sfx.py --force    # regenerate everything

Needs ELEVENLABS_API_KEY in backend/.env. Writes frontend/public/sfx/<name>.mp3, which the game
plays for free from then on (no API calls during play). Missing files are simply silent.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "frontend" / "public" / "sfx"
URL = "https://api.elevenlabs.io/v1/sound-generation?output_format=mp3_44100_128"

# name: (prompt, seconds, loop)
SFX = {
    "heartbeat": ("Close, deep human heartbeat, steady lub-dub, dry, no music, no reverb", 4.0, True),
    "night": ("Quiet night ambience in a small old town, distant wind, crickets, an owl far away, eerie, no music", 20.0, True),
    "card_flip": ("A single playing card flipped over quickly on a felt table, crisp snap", 1.0, False),
    "deal": ("Several playing cards dealt across a wooden table, quick shuffling slides", 2.0, False),
    "body_found": ("Dramatic low orchestral sting with a deep boom, film-noir murder reveal, short", 3.0, False),
    "gavel": ("A single heavy wooden gavel strike in a courtroom, echo", 1.5, False),
    "vote_tension": ("Tense low ticking clock with a soft rising drone, suspense, no melody", 6.0, True),
    "win": ("Short triumphant brass flourish, victory, cinematic", 3.0, False),
    "lose": ("Short descending sad trombone wah-wah, comedic defeat", 3.0, False),
}


def generate(key: str, prompt: str, seconds: float, loop: bool) -> bytes:
    body = {"text": prompt, "duration_seconds": seconds, "prompt_influence": 0.5, "loop": loop}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "xi-api-key": key})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", nargs="*", help="names to (re)generate, e.g. heartbeat night")
    a = ap.parse_args()
    load_dotenv(ROOT / "backend" / ".env")
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        print("Set ELEVENLABS_API_KEY in backend/.env first.")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (prompt, seconds, loop) in SFX.items():
        if a.only and name not in a.only:
            continue
        path = OUT / f"{name}.mp3"
        if path.exists() and not a.force and not a.only:
            print(f"skip {name} (exists)")
            continue
        try:
            path.write_bytes(generate(key, prompt, seconds, loop))
            print(f"saved {path.relative_to(ROOT)}")
        except urllib.error.HTTPError as exc:
            print(f"failed {name}: HTTP {exc.code} {exc.read()[:200]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
