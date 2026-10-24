"""Simulated physiology, so the whole loop runs without a camera or Presage access.

Heart rate and breathing follow a latent arousal level. Arousal rises with the stress of the
moment (a bluff in poker, a lie in Mafia) and after a bad loss (tilt). `leakiness` scales how
much of that shows up: 0 is a perfect poker face.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class SimPhysiology:
    leakiness: float = 1.0
    resting_hr: float = 72.0
    resting_br: float = 14.0
    seed: int | None = None
    arousal: float = 0.0
    tilt: float = 0.0
    _rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def on_hand_result(self, result_bb: float) -> None:
        if result_bb < -5:
            self.tilt = min(self.tilt + 0.6, 2.0)

    def tick(self, t: float, stress: float | None) -> dict[str, float]:
        """Advance one second. `stress` (0 calm .. ~2 very nervous) is scaled by leakiness; None between rounds."""
        target = self.tilt + (self.leakiness * stress if stress is not None else 0.0)
        self.arousal += (target - self.arousal) * 0.35
        self.tilt *= 0.995
        hr = self.resting_hr + 7.0 * self.arousal + self._rng.gauss(0, 1.8) + 1.5 * math.sin(t / 9.0)
        br = self.resting_br + 2.2 * self.arousal + self._rng.gauss(0, 0.6)
        return {"t": t, "hr": round(hr, 1), "br": round(br, 1)}


def poker_stress(strength: float | None, pot_bb: float) -> float | None:
    """Weak hand in a big pot: the urge to bluff."""
    if strength is None:
        return None
    weak = max(0.0, 0.5 - strength) * 4.0      # 0 at 50% equity, 2 at 0%
    stakes = math.log1p(pot_bb) / math.log(40)  # roughly 0..1 over normal pots
    return weak * (0.5 + stakes)
