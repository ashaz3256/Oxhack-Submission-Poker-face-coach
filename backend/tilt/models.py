"""A small online model: logistic regression refit after every labelled answer."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MIN_PER_CLASS = 3


class OnlineLogit:
    """Logistic regression refit on every new label; falls back to a smoothed base rate."""

    def __init__(self, feature_names: list[str], prior_rate: float = 0.3, prior_weight: float = 10.0):
        self.feature_names = feature_names
        self.prior_rate = prior_rate
        self.prior_weight = prior_weight
        self.X: list[list[float]] = []
        self.y: list[int] = []
        self._pipe = None

    def base_rate(self) -> float:
        return (sum(self.y) + self.prior_rate * self.prior_weight) / (len(self.y) + self.prior_weight)

    @property
    def ready(self) -> bool:
        positives = sum(self.y)
        return positives >= MIN_PER_CLASS and len(self.y) - positives >= MIN_PER_CLASS

    def add(self, x: list[float], label: bool) -> None:
        self.X.append(x)
        self.y.append(int(label))
        if self.feature_names and self.ready:
            self._pipe = make_pipeline(StandardScaler(), LogisticRegression(C=0.5))
            self._pipe.fit(np.array(self.X), np.array(self.y))

    def prior(self, offset: float = 0.0) -> float:
        """Base rate, optionally shifted by a log-odds offset from a pretrained model."""
        b = min(max(self.base_rate(), 1e-6), 1 - 1e-6)
        return float(1 / (1 + np.exp(-(np.log(b / (1 - b)) + offset))))

    def predict(self, x: list[float], prior_offset: float = 0.0) -> float:
        prior = self.prior(prior_offset)
        if self._pipe is None:
            return prior
        p = float(self._pipe.predict_proba(np.array([x]))[0, 1])
        # Shrink toward the prior while the model has little data.
        w = len(self.y) / (len(self.y) + self.prior_weight)
        return w * p + (1 - w) * prior

    def contributions(self, x: list[float]) -> dict[str, float]:
        """How much each feature pushed this prediction, in log-odds."""
        if self._pipe is None:
            return {}
        scaled = self._pipe[0].transform(np.array([x]))[0]
        coef = self._pipe[-1].coef_[0]
        return {n: float(c * v) for n, c, v in zip(self.feature_names, coef, scaled)}

    def coefficients(self) -> dict[str, float]:
        if self._pipe is None:
            return {}
        coef = self._pipe[-1].coef_[0]
        return {n: round(float(c), 3) for n, c in zip(self.feature_names, coef)}


def safe_auc(y: list[int], p: list[float]) -> float | None:
    if len(set(y)) < 2:
        return None
    return round(float(roc_auc_score(y, p)), 3)
