"""A simulated player for tests and the demo report, before real games exist.

Their face and voice drift from baseline when they lie, scaled by `leakiness` (0 = perfect
poker face). The shifts follow common findings (higher pitch, more pauses, gaze aversion,
lip pressing) but the sizes are made up: real numbers come from real games.
"""
from __future__ import annotations

import random

from .mafia import BOTS, MafiaSession

BASE = {"pitch": 140.0, "pitch_var": 18.0, "loudness": 0.05, "rate": 2.6, "pause": 0.2, "latency": 1.2,
        "blink": 17.0, "brow_raise": 0.12, "brow_down": 0.05, "lip_press": 0.1, "gaze_away": 0.15,
        "head_motion": 0.02, "smile": 0.2, "hr": 74.0}
NOISE = {"pitch": 10.0, "pitch_var": 5.0, "loudness": 0.012, "rate": 0.4, "pause": 0.06, "latency": 0.5,
         "blink": 5.0, "brow_raise": 0.05, "brow_down": 0.03, "lip_press": 0.04, "gaze_away": 0.05,
         "head_motion": 0.006, "smile": 0.08, "hr": 4.0}
LIE_SHIFT = {"pitch": 12.0, "pause": 0.08, "latency": 0.7, "blink": 6.0, "gaze_away": 0.08, "lip_press": 0.06,
             "hr": 6.0, "rate": -0.3, "smile": -0.06}
TRUTHFUL = ["I was at home all night, honestly nothing happened.", "I think it's Ben, he keeps changing his story.",
            "No, I'm a villager. Ask me anything.", "I trust Cleo, she's been consistent all game."]
LYING = ["No, I'm not the mafia, I was asleep the whole night.", "Um, I think it's maybe Dev, he's been weird.",
         "I swear I'm just a villager, trust me.", "I was at home, I didn't go anywhere, honestly."]


class FakeClock:
    def __init__(self, start: float = 1_700_000_000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def raw_features(rng: random.Random, lying: bool, leakiness: float) -> dict:
    out = {}
    for k, base in BASE.items():
        shift = LIE_SHIFT.get(k, 0.0) * leakiness if lying else 0.0
        out[k] = max(0.0, base + shift + rng.gauss(0, NOISE[k]))
    return out


def play(n_games: int = 12, leakiness: float = 1.0, seed: int = 7) -> MafiaSession:
    clock = FakeClock()
    rng = random.Random(seed)
    s = MafiaSession(seed=seed, clock=clock)
    for i in range(6):
        lie = i % 2 == 1  # practice round: truth, lie, truth, lie, truth, lie
        text = rng.choice(LYING if lie else TRUTHFUL)
        s.calibrate(text, raw_features(rng, lie, leakiness), is_lie=lie)
    for _ in range(n_games):
        game = s.new_game()
        while game.phase != "over":
            clock.advance(rng.randint(3, 9))
            kind = game.awaiting["kind"]
            if kind == "pick":
                s.speak(rng.choice(["one", "two", "three", "four", "five", "six"]))
            elif kind == "answer":
                lying = game.you_mafia
                s.speak(rng.choice(LYING if lying else TRUTHFUL), raw_features(rng, lying, leakiness))
            else:
                s.speak(f"I vote {rng.choice([b for b in BOTS if game.alive[b]])}")
        s.view()  # drain lines
    return s
