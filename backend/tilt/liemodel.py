"""The lie detector: a personal logistic model that starts from research-based cue directions.

Small data is the whole problem here: one player gives a few dozen answers at most. So instead
of learning from zero, the model starts from a prior over which way each cue points when people
lie (meta-analyses of deception cues: higher pitch, longer pauses and response latency, pressed
lips, fewer details, more hedging and denial) plus a language judge (Gemini reading the answer).
Each labelled answer, starting with the practice round of instructed truths and lies, pulls the
weights toward *your* tells. Fit is MAP: logistic loss + a Gaussian prior centred on those
research weights, solved with Newton's method.
"""
from __future__ import annotations

import math

import numpy as np

from .cues import LIE_FEATURES, Baseline, feature_dict

FEATURES = LIE_FEATURES + ["llm"]

# Prior weights per unit of feature (z-scores for face and voice, rescaled word rates, logit of the
# language judge). Signs follow the deception literature; sizes are deliberately modest.
PRIOR = {
    "pitch_z": 0.35, "pitch_var_z": 0.1, "loudness_z": 0.0, "rate_z": -0.1, "pause_z": 0.3, "latency_z": 0.35,
    "blink_z": 0.15, "brow_raise_z": 0.1, "brow_down_z": 0.05, "lip_press_z": 0.3, "gaze_away_z": 0.15,
    "head_motion_z": -0.1, "smile_z": -0.05, "hr_z": 0.25, "rr_z": 0.2,
    "hedges": 0.35, "fillers": 0.25, "negations": 0.25, "log_words": -0.2, "llm": 0.9,
    # Log-odds from the face model trained on the I3D dataset (research/); 0 when none is installed.
    "i3d_face": 1.0,
}
# Word cues are rates per word; scale them so one unit is a meaningful change.
SCALE = {"hedges": 12.0, "fillers": 12.0, "negations": 8.0, "log_words": 1.0}
CENTRE = {"log_words": 2.3}
PRIOR_STRENGTH = 2.0   # how hard the prior pulls; lower = learns your quirks faster


def _vector(f: dict) -> np.ndarray:
    return np.array([(f.get(k, 0.0) - CENTRE.get(k, 0.0)) * SCALE.get(k, 1.0) for k in FEATURES])


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class LieModel:
    def __init__(self, prior_strength: float = PRIOR_STRENGTH):
        self.baseline = Baseline()
        self.prior_strength = prior_strength
        self.w0 = np.array([PRIOR[k] for k in FEATURES])
        self.w = self.w0.copy()
        self.b = 0.0
        # Each sample: raw measurements, transcript, language judge score, label.
        self.samples: list[dict] = []

    # ---- data ---------------------------------------------------------------------------------
    def features(self, raw: dict, transcript: str, llm: float | None) -> dict:
        f = feature_dict(self.baseline, raw, transcript)
        f["i3d_face"] = float(raw.get("i3d_face") or 0.0)
        f["llm"] = 0.0 if llm is None else math.log(max(llm, 0.02) / max(1 - llm, 0.02))
        return f

    def add(self, raw: dict, transcript: str, llm: float | None, label: bool, honest_baseline: bool = False) -> None:
        if honest_baseline:
            self.baseline.add(raw)
        self.samples.append({"raw": raw, "transcript": transcript, "llm": llm, "y": int(label)})
        self.fit()

    def _design(self, samples: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        X = np.array([_vector(self.features(s["raw"], s["transcript"], s["llm"])) for s in samples])
        y = np.array([s["y"] for s in samples], dtype=float)
        return X, y

    # ---- fitting ------------------------------------------------------------------------------
    def fit(self) -> None:
        self.w, self.b = self._fit(self.samples)

    def _fit(self, samples: list[dict]) -> tuple[np.ndarray, float]:
        if not samples:
            return self.w0.copy(), 0.0
        X, y = self._design(samples)
        n, d = X.shape
        A = np.hstack([X, np.ones((n, 1))])
        theta = np.concatenate([self.w0, [0.0]])
        prior = np.concatenate([self.w0, [0.0]])
        lam = np.full(d + 1, self.prior_strength)
        lam[-1] = 0.1  # the intercept (your base lie rate) is free to move
        for _ in range(25):
            p = _sigmoid(A @ theta)
            grad = A.T @ (p - y) + lam * (theta - prior)
            H = (A * (p * (1 - p))[:, None]).T @ A + np.diag(lam)
            step = np.linalg.solve(H, grad)
            theta -= step
            if np.abs(step).max() < 1e-6:
                break
        return theta[:-1], float(theta[-1])

    # ---- reading ------------------------------------------------------------------------------
    def predict(self, raw: dict, transcript: str, llm: float | None) -> tuple[float, dict[str, float], dict]:
        f = self.features(raw, transcript, llm)
        x = _vector(f)
        p = float(_sigmoid(x @ self.w + self.b))
        contrib = {k: float(v) for k, v in zip(FEATURES, x * self.w)}
        return p, contrib, f

    def cross_validated_accuracy(self) -> float | None:
        """Leave-one-out accuracy over the labelled answers so far (each scored by a model that never saw it)."""
        ys = [s["y"] for s in self.samples]
        if len(self.samples) < 4 or len(set(ys)) < 2:
            return None
        hits = 0
        for i, s in enumerate(self.samples):
            w, b = self._fit(self.samples[:i] + self.samples[i + 1:])
            x = _vector(self.features(s["raw"], s["transcript"], s["llm"]))
            hits += int((float(_sigmoid(x @ w + b)) > 0.5) == bool(s["y"]))
        return hits / len(self.samples)

    def coefficients(self) -> dict[str, float]:
        return {k: round(float(v), 3) for k, v in zip(FEATURES, self.w)}

    @property
    def n_lies(self) -> int:
        return sum(s["y"] for s in self.samples)

    @property
    def base_rate(self) -> float:
        ys = [s["y"] for s in self.samples]
        return (sum(ys) + 4) / (len(ys) + 10)
