"""Turn a raw vitals stream into features for the models."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Features the tell model (Shark) and the tilt model read. Order matters: it is the column order.
TELL_FEATURES = ["hr_z", "hr_slope", "br_z", "decision_s"]
TILT_FEATURES = TELL_FEATURES + ["last_result_bb", "drawdown_bb"]

BASELINE_SECONDS = 45.0
WINDOW_SECONDS = 10.0
MIN_HR_STD = 3.0
MIN_BR_STD = 1.0


@dataclass
class Sample:
    t: float
    hr: float
    br: float | None = None


def _baseline(samples: list[Sample]) -> tuple[float, float, float, float]:
    start = samples[0].t
    base = [s for s in samples if s.t - start <= BASELINE_SECONDS] or samples
    hr = np.array([s.hr for s in base])
    br = np.array([s.br for s in base if s.br is not None])
    hr_mu, hr_sd = float(hr.mean()), max(float(hr.std()), MIN_HR_STD)
    br_mu, br_sd = (float(br.mean()), max(float(br.std()), MIN_BR_STD)) if len(br) else (15.0, 2.0)
    return hr_mu, hr_sd, br_mu, br_sd


def vitals_features(samples: list[Sample], at: float) -> dict[str, float]:
    """Vitals in the WINDOW_SECONDS before `at`, as z-scores against the session's opening baseline."""
    past = [s for s in samples if s.t <= at]
    if not past:
        return {"hr_z": 0.0, "hr_slope": 0.0, "br_z": 0.0, "hr": float("nan")}
    hr_mu, hr_sd, br_mu, br_sd = _baseline(past)
    window = [s for s in past if at - s.t <= WINDOW_SECONDS] or past[-1:]
    hr = np.array([s.hr for s in window])
    t = np.array([s.t for s in window])
    slope = float(np.polyfit(t - t[0], hr, 1)[0]) if len(window) >= 3 and t[-1] > t[0] else 0.0
    brs = [s.br for s in window if s.br is not None]
    br_z = (float(np.mean(brs)) - br_mu) / br_sd if brs else 0.0
    return {
        "hr_z": (float(hr.mean()) - hr_mu) / hr_sd,
        "hr_slope": slope,
        "br_z": br_z,
        "hr": float(hr.mean()),
    }


def vector(features: dict[str, float], names: list[str]) -> list[float]:
    return [float(features.get(n, 0.0)) for n in names]
