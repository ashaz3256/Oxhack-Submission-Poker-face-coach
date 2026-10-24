"""Deception cues for one spoken answer: face and voice (measured in the browser) plus the words."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

# Raw per-answer measurements sent by the browser, and a typical spread for each when we
# have too little of your own baseline to estimate it.
RAW = {
    "pitch": 20.0,        # median voice pitch, Hz
    "pitch_var": 10.0,    # pitch standard deviation, Hz
    "loudness": 0.02,     # mean RMS while voiced
    "rate": 0.6,          # words per voiced second
    "pause": 0.1,         # silent share of the answer
    "latency": 1.0,       # seconds from the question to your first word
    "blink": 8.0,         # blinks per minute
    "brow_raise": 0.08,   # browInnerUp blendshape
    "brow_down": 0.08,    # browDown blendshapes
    "lip_press": 0.08,    # mouthPress / mouthRoll blendshapes
    "gaze_away": 0.08,    # eyes looking out/down/up
    "head_motion": 0.01,  # spread of nose-tip position
    "smile": 0.1,         # mouthSmile blendshapes
    "hr": 6.0,            # heart rate from the webcam, bpm
}
TEXT = ["hedges", "fillers", "negations", "log_words"]
LIE_FEATURES = [f"{k}_z" for k in RAW] + TEXT

LABELS = {
    "pitch_z": "voice pitch", "pitch_var_z": "pitch wobble", "loudness_z": "loudness", "rate_z": "talking speed",
    "pause_z": "pauses", "latency_z": "slow to start", "blink_z": "blinking", "brow_raise_z": "raised eyebrows",
    "brow_down_z": "frowning", "lip_press_z": "pressed lips", "gaze_away_z": "looking away",
    "head_motion_z": "head movement", "smile_z": "smiling", "hr_z": "heart rate", "hedges": "hedging words",
    "fillers": "ums and ahs", "negations": "denials", "log_words": "answer length",
}

FILLERS = {"um", "umm", "uh", "uhh", "er", "erm", "hmm", "like"}
HEDGES = ["maybe", "probably", "i think", "kind of", "sort of", "i guess", "honestly", "to be honest",
          "believe me", "trust me", "i swear", "literally", "actually", "basically", "i mean", "you know"]
NEGATION = re.compile(r"\b(no|not|never|nothing|nobody|wasn't|didn't|don't|isn't|haven't|won't|can't|n't)\b")


def words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def text_cues(transcript: str) -> dict[str, float]:
    w = words(transcript)
    n = max(len(w), 1)
    low = " " + " ".join(w) + " "
    hedges = sum(low.count(f" {h} ") for h in HEDGES)
    return {
        "hedges": hedges / n,
        "fillers": sum(x in FILLERS for x in w) / n,
        "negations": len(NEGATION.findall(low)) / n,
        "log_words": math.log1p(len(w)),
    }


@dataclass
class Baseline:
    """Your own calm, truthful self. Every answer is measured against it."""

    values: dict[str, list[float]] = field(default_factory=lambda: {k: [] for k in RAW})

    def add(self, raw: dict) -> None:
        for k in RAW:
            v = raw.get(k)
            if v is not None and v == v:
                self.values[k].append(float(v))

    def z(self, raw: dict) -> dict[str, float]:
        out = {}
        for k, default in RAW.items():
            v = raw.get(k)
            hist = self.values[k]
            if v is None or v != v or not hist:
                out[f"{k}_z"] = 0.0
                continue
            mean = sum(hist) / len(hist)
            if len(hist) >= 3:
                sd = math.sqrt(sum((h - mean) ** 2 for h in hist) / (len(hist) - 1))
                scale = max(sd, default * 0.5)
            else:
                scale = default
            out[f"{k}_z"] = max(-5.0, min(5.0, (float(v) - mean) / scale))
        return out


def feature_dict(baseline: Baseline, raw: dict, transcript: str) -> dict[str, float]:
    return {**baseline.z(raw), **text_cues(transcript)}


def describe(name: str, value: float) -> str:
    label = LABELS.get(name, name)
    if name.endswith("_z"):
        return f"{label} {'up' if value > 0 else 'down'} {abs(value):.1f}σ"
    return label
