"""Presage readings: per-answer windows, confidence filtering and the API. Readings here are made up."""
from fastapi.testclient import TestClient

from tilt.presage import PresageFeed


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def test_window_averages_and_ignores_low_confidence():
    clock = Clock()
    feed = PresageFeed(clock)
    feed.add([{"t": 990, "pulse": 70, "pulse_conf": 0.9, "br": 14, "br_conf": 0.9, "hrv_rmssd": 50},
              {"t": 995, "pulse": 80, "pulse_conf": 0.9, "br": 16, "br_conf": 0.9, "hrv_rmssd": 40},
              {"t": 996, "pulse": 150, "pulse_conf": 0.1, "source": "presage"},
              {"t": 900, "pulse": 999, "pulse_conf": 1.0}])
    w = feed.window(989, 1000)
    assert w == {"hr": 75, "br": 15, "hrv": 45, "_source": "presage"}
    assert feed.window(1001, 1010) == {}


def test_status_goes_stale():
    clock = Clock()
    feed = PresageFeed(clock)
    feed.add([{"t": 999, "pulse": 72, "source": "mock"}])
    assert feed.status()["connected"] and feed.status()["source"] == "mock"
    clock.t = 1010
    assert not feed.status()["connected"]


def test_presage_values_reach_the_answer_features():
    from tilt.mafia import MafiaSession

    clock = Clock()
    s = MafiaSession(seed=1, clock=clock)
    s.presage = PresageFeed(clock)
    s.presage.add([{"t": 995, "pulse": 88, "pulse_conf": 0.9, "br": 18, "br_conf": 0.9, "hrv_rmssd": 30,
                    "source": "presage"}])
    s.calibrate("I went to work and then I cooked dinner at home.", {"hr": 60})
    u = s.utterances[-1]
    assert u.raw["hr"] == 88 and u.raw["br"] == 18 and u.raw["hrv"] == 30 and u.raw["vitals_source"] == "presage"
    assert "br_z" in u.features and "hrv_z" in u.features


def test_api_accepts_samples():
    from tilt.api import app

    c = TestClient(app)
    assert c.post("/api/presage", json={"samples": [{"t": 1.0, "pulse": 70}, {"bad": 1}]}).json() == {"stored": 1}
    assert "connected" in c.get("/api/presage").json()
