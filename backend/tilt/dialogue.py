"""What the AI players say. Gemini writes it when a key is set; otherwise varied templates do."""
from __future__ import annotations

import json
import os
import random
import urllib.request

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

PERSONAS = {
    "Ava": "sharp ex-lawyer, direct, cross-examines people, short punchy sentences",
    "Ben": "laid-back joker, teases people, casual slang, secretly observant",
    "Cleo": "anxious overthinker, worries out loud, quick to panic and accuse",
    "Dev": "calm analyst, talks in probabilities and evidence, dry humour",
    "Eli": "theatrical old storyteller, dramatic, suspicious of everyone",
}

QUESTIONS = [
    "Where were you last night?",
    "Are you the mafia? Look at us and answer.",
    "Who do you think it is, and why?",
    "You've been quiet. What are you hiding?",
    "Do you trust {other}? Why?",
    "If you were the mafia, who would you have killed first?",
    "Tell us exactly what you did after dark.",
]
SUSPECT = [
    "My money's on {x}.", "{x} has been way too quiet.", "I don't trust {x}. Not one bit.",
    "Something about {x} feels off to me.", "Did anyone else notice {x} dodging questions?",
]
OPENERS = [
    "Alright, someone here is lying. Let's find them.", "Nobody leaves until we know who did it.",
    "Let's keep this civil. Mostly.", "I've got a bad feeling about today.",
]
CUE_CALLOUTS = {
    "pitch_z": "Your voice went up a notch there.",
    "pitch_var_z": "Your voice was wobbling on that one.",
    "rate_z": "You sped right up when you said that.",
    "pause_z": "That was a lot of pauses for a simple answer.",
    "latency_z": "You took your time getting started.",
    "blink_z": "You're blinking a lot all of a sudden.",
    "gaze_away_z": "You couldn't look at us when you said that.",
    "lip_press_z": "You pressed your lips together right after.",
    "brow_raise_z": "Your eyebrows shot up there.",
    "brow_down_z": "Why the frown?",
    "hr_z": "I can practically hear your heart racing.",
    "hedges": "So many 'honestly's. Hmm.",
    "fillers": "Lot of 'ums' in that answer.",
    "head_motion_z": "You're fidgeting.",
}
CALM = ["Okay. That sounded honest to me.", "Fair enough, I believe you. For now.", "Hm. Steady answer.",
        "That checks out, I think."]
DOUBT = ["I'm not buying it.", "That doesn't add up.", "Convenient.", "Interesting answer. Too interesting."]


def _gemini(prompt: str) -> list[dict] | None:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.9}}
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "x-goog-api-key": key})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = json.loads(resp.read())
        lines = json.loads(raw["candidates"][0]["content"]["parts"][0]["text"])
        return [l for l in lines if isinstance(l, dict) and l.get("speaker") and l.get("text")]
    except Exception:
        return None


def ai_lines(*, speakers: list[str], task: str, game: dict, rng: random.Random, fallback) -> list[dict]:
    """Lines for `speakers`, in order. `game` is the public context; `fallback` builds template lines."""
    prompt = (
        "You write dialogue for AI players in a spoken game of Mafia with one human. Stay in character, "
        "sound natural when read aloud, max 22 words per line, no stage directions, no emoji.\n"
        f"Players and personas: {json.dumps({s: PERSONAS[s] for s in speakers})}\n"
        f"Game state (secret roles are only known to the mafia player, never reveal them): {json.dumps(game)}\n"
        f"Task: {task}\n"
        f'Return a JSON list like [{{"speaker": "{speakers[0]}", "text": "..."}}] with exactly one line for each of '
        f"{speakers}, in that order."
    )
    lines = _gemini(prompt)
    if lines and [l["speaker"] for l in lines] == speakers:
        return [{"speaker": l["speaker"], "text": str(l["text"]).strip()} for l in lines]
    return fallback()


NIGHT_QUESTIONS = {QUESTIONS[0], QUESTIONS[6]}


def template_question(rng: random.Random, others: list[str], day: int = 2) -> str:
    bank = QUESTIONS if day > 1 else [q for q in QUESTIONS if q not in NIGHT_QUESTIONS]
    return rng.choice(bank).format(other=rng.choice(others))


def template_reaction(rng: random.Random, p_lie: float, top_cue: str | None) -> str:
    if p_lie > 0.6 and top_cue in CUE_CALLOUTS:
        return CUE_CALLOUTS[top_cue] + " " + rng.choice(DOUBT)
    if p_lie > 0.55:
        return rng.choice(DOUBT)
    return rng.choice(CALM)
