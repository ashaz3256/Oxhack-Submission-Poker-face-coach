"""Vitals from the Presage SmartSpectra sidecar (presage/presage.mjs), summarised per answer.

The sidecar posts a reading about once a second to /api/presage. For each answer we average the
readings inside that answer's time window. Presage pulse replaces the in-browser webcam pulse when
its confidence is good; breathing rate and HRV are new signals the browser cannot measure.
Samples tagged source="mock" (the sidecar's --mock mode) are kept apart and labelled as such.
"""
from __future__ import annotations

import time
from collections import deque
from statistics import fmean

MIN_CONFIDENCE = 0.5
MAX_AGE_S = 600
STALE_S = 5.0          # no reading for this long means the sidecar is not running


class PresageFeed:
    def __init__(self, clock=time.time):
        self.clock = clock
        self.samples: deque[dict] = deque()

    def add(self, samples: list[dict]) -> int:
        n = 0
        for s in samples:
            if not isinstance(s.get("t"), (int, float)):
                continue
            self.samples.append(dict(s))
            n += 1
        cutoff = self.clock() - MAX_AGE_S
        while self.samples and self.samples[0]["t"] < cutoff:
            self.samples.popleft()
        return n

    def status(self) -> dict:
        latest = self.samples[-1] if self.samples else None
        live = bool(latest) and self.clock() - latest["t"] < STALE_S
        return {"connected": live, "source": latest.get("source") if latest else None,
                "pulse": latest.get("pulse") if live else None, "br": latest.get("br") if live else None,
                "hrv_rmssd": latest.get("hrv_rmssd") if live else None}

    def window(self, start: float, end: float) -> dict:
        """Averages over [start, end] as extra raw cues: hr (if confident), br, hrv. Empty if no data."""
        rows = [s for s in self.samples if start <= s["t"] <= end]
        if not rows:
            return {}

        def avg(key, conf=None):
            vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))
                    and (conf is None or (r.get(conf) or 0) >= MIN_CONFIDENCE)]
            return fmean(vals) if vals else None

        out = {"br": avg("br", "br_conf"), "hrv": avg("hrv_rmssd")}
        hr = avg("pulse", "pulse_conf")
        if hr is not None:
            out["hr"] = hr
        out = {k: v for k, v in out.items() if v is not None}
        if out:
            out["_source"] = rows[-1].get("source", "presage")
        return out


FEED = PresageFeed()
