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
    "Okay so where were you last night? Be specific.",
    "Hang on. Are you the mafia? Look at us and say it.",
    "Alright, who do you reckon it is, and why?",
    "You've gone really quiet. What's going on with you?",
    "Do you actually trust {other}? Why?",
    "If you were the mafia, who'd you have gone for first?",
    "Walk us through what you did after dark. Step by step.",
]
SUSPECT = [
    "Honestly? My money's on {x}.", "{x}'s been way too quiet, no?", "I don't trust {x}. Not even a bit.",
    "Something about {x} feels off, I can't put my finger on it.", "Did nobody else see {x} dodge that?",
]
OPENERS = [
    "Right, someone here's lying. Let's just find them.", "Okay nobody's leaving till we know who did it.",
    "Let's keep this civil. Ish. Ha.", "I've got a really bad feeling about today, guys.",
]
CUE_CALLOUTS = {
    "pitch_z": "Wait, your voice just went right up there.",
    "pitch_var_z": "Your voice was all wobbly on that one.",
    "rate_z": "You sped up a lot just then, you know that?",
    "pause_z": "That's a lot of pauses for such a simple question.",
    "latency_z": "You took ages to start answering that.",
    "blink_z": "Why are you blinking so much all of a sudden?",
    "gaze_away_z": "You literally couldn't look at us when you said that.",
    "lip_press_z": "You pressed your lips together right after, I saw that.",
    "brow_raise_z": "Your eyebrows shot right up there, ha.",
    "brow_down_z": "Why the frown though?",
    "hr_z": "Mate, I can practically hear your heart going.",
    "hedges": "That's a lot of 'honestly' for someone being honest.",
    "fillers": "So many 'ums' in that answer.",
    "head_motion_z": "You're fidgeting loads.",
    "llm": "That story's got holes in it.",
}
CALM = ["Okay, yeah. That sounded honest to me.", "Fair enough. I believe you. For now.", "Hm. Pretty steady answer.",
        "Yeah, that checks out I think."]
DOUBT = ["Nah, I'm not buying it.", "That doesn't add up though.", "Bit convenient, isn't it?",
         "Hmm. Interesting answer. Too interesting."]


def _gemini_json(prompt: str, timeout: float = 8):
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.2}}
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "x-goog-api-key": key})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = json.loads(resp.read())
        return json.loads(raw["candidates"][0]["content"]["parts"][0]["text"])
    except Exception:
        return None


def judge_lie(question: str, answer: str, recent: list[str]) -> float | None:
    """A language-only read: how deceptive does this answer sound, judged from the words alone?

    The judge never sees the player's role. It looks for the verbal cues people use when they
    lie: vague or missing detail, over-insistence on honesty, deflection, inconsistency with what
    they said earlier, implausible content, and answers that dodge the question.
    """
    if not answer.strip():
        return None
    prompt = (
        "You are a forensic linguist scoring one spoken answer for deception. You do NOT know if the speaker "
        "is lying. Judge only the words: vagueness or lack of concrete detail, over-insisting on honesty "
        "('honestly', 'I swear'), deflecting or dodging the question, contradicting earlier statements, "
        "implausible or made-up-sounding content, distancing language. Plain, specific, direct answers "
        "score low.\n"
        f"Earlier conversation: {json.dumps(recent[-10:])}\n"
        f"Question: {question!r}\nAnswer (speech-to-text, may have small errors): {answer!r}\n"
        'Return JSON {"deception": number from 0 to 1, "reason": "few words"}.'
    )
    out = _gemini_json(prompt)
    try:
        return max(0.0, min(1.0, float(out["deception"])))
    except Exception:
        return None


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
        "You write dialogue for players in a spoken game of Mafia with one human, played on a voice call. "
        "Write exactly how real friends talk out loud: contractions, casual phrasing, short sentences, the odd "
        "'wait', 'hang on', 'okay so', 'mate', a laugh written as 'ha', reacting to each other by name. Never "
        "sound like an assistant or a narrator. Max 20 words per line, no stage directions, no emoji.\n"
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
    if p_lie > 0.55 and top_cue in CUE_CALLOUTS:
        return CUE_CALLOUTS[top_cue] + " " + rng.choice(DOUBT)
    if p_lie > 0.55:
        return rng.choice(DOUBT)
    return rng.choice(CALM)
