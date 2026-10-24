"""The face model trained on the published I3D deception dataset, applied to the player's webcam.

`research/train_face_model.py` writes `pretrained_face_model.json` only when the model beats
chance on held-out participants (95% interval above AUC 0.5). If the file is missing, the score
is 0 for every answer, so the game behaves exactly as before, and `status()` says so; the UI shows
that as demonstration mode rather than pretending a trained model exists.

Each turn's face features are centred on the player's own running mean, as the training data was
centred on each participant's own mean. Response latency is NOT part of this model (the dataset
has no question-to-answer timing); it stays in the game's online model as `latency_z`.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

from .face_features import PRIMITIVES, window_features

MODEL_PATH = Path(os.getenv("TILT_FACE_MODEL", Path(__file__).with_name("pretrained_face_model.json")))
CLIP = 4.0  # keep one odd turn from dominating the log-odds


def _load(path: Path) -> dict | None:
    try:
        model = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    need = {"features", "scaler_mean", "scaler_scale", "coef", "intercept", "eval", "source"}
    return model if need <= set(model) else None


class FaceScorer:
    """Per-session scorer: holds the player's running face baseline."""

    def __init__(self, model: dict | None = None, path: Path = MODEL_PATH):
        self.model = model if model is not None else _load(path)
        self.history: list[dict[str, float]] = []

    @property
    def available(self) -> bool:
        return self.model is not None

    def status(self) -> dict:
        if not self.model:
            return {"mode": "demo", "trained": False,
                    "note": "No dataset-trained face model is installed; face score is 0 (demonstration mode)."}
        e = self.model["eval"]
        return {"mode": "trained", "trained": True, "auc": e["auc"], "auc_95ci": e["auc_95ci"], "brier": e["brier"],
                "participants": e["participants"], "recordings": e["recordings"], "source": self.model["source"]}

    def score(self, frames: list[list[float]] | None) -> tuple[float, dict | None]:
        """Log-odds shift from this turn's face (no intercept), and the turn's window features."""
        if not frames:
            return 0.0, None
        rows = [[float(v) if v is not None else float("nan") for v in f] for f in frames if len(f) == len(PRIMITIVES)]
        feats = window_features([dict(zip(PRIMITIVES, r)) for r in rows])
        if not math.isfinite(feats.get("frames", 0)) or feats["frames"] < 10:
            return 0.0, None
        names = self.model["features"] if self.model else [k for k in feats if k != "frames"]
        vals = {k: feats.get(k, float("nan")) for k in names}
        if self.history:
            mean = {k: np.nanmean([h[k] for h in self.history]) if any(h[k] == h[k] for h in self.history) else np.nan
                    for k in names}
        else:
            mean = vals  # first turn: nothing to compare with yet, so it reads as typical
        centred = {k: (vals[k] - mean[k]) if (vals[k] == vals[k] and mean[k] == mean[k]) else 0.0 for k in names}
        self.history.append(vals)
        if not self.model:
            return 0.0, feats
        m = self.model
        z = (np.array([centred[k] for k in names]) - np.array(m["scaler_mean"])) / np.array(m["scaler_scale"])
        s = float(np.dot(np.array(m["coef"]), z))
        return max(-CLIP, min(CLIP, s)), feats
