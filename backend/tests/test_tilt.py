import os

os.environ["DATABASE_URL"] = "sqlite://"
os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("ELEVENLABS_API_KEY", None)

from fastapi.testclient import TestClient  # noqa: E402

from tilt.api import app  # noqa: E402
from tilt.cues import Baseline, text_cues  # noqa: E402
from tilt.language import last_name, parse_number  # noqa: E402
from tilt.report import report  # noqa: E402
from tilt.synthetic import play  # noqa: E402


def test_language_parsing():
    assert parse_number("um I'll take number four") == 4
    assert parse_number("the third one") == 3
    assert last_name("not Ben, I vote Chloe", ["Ben", "Cleo", "Dev"]) == "Cleo"
    assert last_name("nobody", ["Ben"]) is None


def test_text_cues():
    c = text_cues("Um, honestly, I swear I didn't do it")
    assert c["fillers"] > 0 and c["hedges"] > 0 and c["negations"] > 0


def test_baseline_zscores():
    b = Baseline()
    for p in (140, 150, 130):
        b.add({"pitch": p})
    assert b.z({"pitch": 160})["pitch_z"] > 1
    assert b.z({})["blink_z"] == 0.0


def test_leaky_liar_is_caught_more_when_read():
    r = report(play(n_games=20, leakiness=1.5, seed=2))
    assert r["lie_auc"] > 0.8
    assert r["caught_by_reader"] > r["caught_by_blind"]


def test_api_plays_a_whole_game_by_voice():
    client = TestClient(app)
    sid = client.post("/api/sessions", json={}).json()["session_id"]
    out = client.post(f"/api/sessions/{sid}/calibrate", json={"transcript": "hi I had a good day", "raw": {"pitch": 140}}).json()
    assert 0 <= out["p_lie"] <= 1
    client.post(f"/api/sessions/{sid}/calibrate", json={"transcript": "I once met the king, honestly", "raw": {"pitch": 170}, "is_lie": True})
    state = client.post(f"/api/sessions/{sid}/games").json()
    assert state["game"]["awaiting"]["kind"] == "pick"
    for _ in range(60):
        kind = state["game"]["awaiting"]["kind"]
        if kind == "again":
            break
        said = {"pick": "number two", "answer": "No I was asleep at home all night",
                "vote": "I vote Ava", "kill": "Ben"}[kind]
        state = client.post(f"/api/sessions/{sid}/speak", json={"transcript": said, "raw": {"pitch": 150}}).json()
        assert state["lines"], kind
    assert state["game"]["phase"] == "over"
    r = client.get(f"/api/sessions/{sid}/report").json()
    assert r["games"] == 1 and r["headline"]
    assert client.post(f"/api/sessions/{sid}/review").json()["text"]
    assert client.post("/api/tts", json={"text": "hi", "speaker": "Ava"}).status_code == 204


def test_practice_round_teaches_the_detector():
    from tilt.liemodel import LieModel
    from tilt.synthetic import raw_features
    import random

    rng = random.Random(0)
    m = LieModel()
    for i in range(10):
        lie = i % 2 == 1
        m.add(raw_features(rng, lie, 1.0), "I was at home" if not lie else "um honestly I was at home", None, lie,
              honest_baseline=not lie)
    p_lie, _, _ = m.predict(raw_features(rng, True, 1.0), "um honestly I swear", None)
    p_truth, _, _ = m.predict(raw_features(rng, False, 1.0), "I was at home", None)
    assert p_lie > 0.6 > p_truth
    assert m.cross_validated_accuracy() >= 0.8


def test_breakdown_marks_right_and_wrong():
    from tilt.breakdown import breakdown

    s = play(n_games=4, leakiness=1.0, seed=3)
    for g in s.games:
        b = breakdown(s, g.number)
        assert b["over"] and b["role"] in ("mafia", "villager")
        kinds = {m["type"] for m in b["moments"]}
        assert "answer" in kinds and "vote" in kinds
        assert all(m["verdict"] in ("good", "bad", "neutral") for m in b["moments"])
        assert len(b["right"]) + len(b["wrong"]) > 0
    mafia_games = [g for g in s.games if g.you_mafia]
    if mafia_games:
        assert breakdown(s, mafia_games[0].number)["tells"], "a leaky liar should have tells"


def test_presage_reading_feeds_the_model():
    import httpx

    from tilt.presage import PresageVitals, parse

    assert parse({"hr": {"0": 70, "5": 72, "10": 74}, "rr": {"0": {"value": 14.0}}}) == {"hr": 72.0, "rr": 14.0}

    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        assert request.headers.get("x-api-key") in (None, "k")
        if request.url.path == "/v1/upload-url":
            return httpx.Response(200, json={"id": "v1", "upload_id": "u1", "urls": ["https://s3.test/part1"]})
        if request.url.host == "s3.test":
            return httpx.Response(200, headers={"ETag": "e1"})
        if request.url.path == "/v1/complete":
            return httpx.Response(200, json={})
        if request.url.path == "/retrieve-data":
            if calls.count("/retrieve-data") < 2:
                return httpx.Response(201)
            return httpx.Response(200, json={"hr": {"1": 96.0, "2": 98.0}, "rr": {"1": 19.0}})
        return httpx.Response(404)

    import tilt.presage as presage_mod

    presage_mod._sleep = lambda _s: None
    v = PresageVitals(key="k", client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert v.submit(b"fake-mp4-bytes", "mp4", background=False)
    assert v.view()["status"] == "ok" and v.hr == 97.0 and v.rr == 19.0
    assert v.fresh() == {"hr": 97.0, "rr": 19.0}

    from tilt.mafia import MafiaSession

    s = MafiaSession(seed=1)
    s.presage = v
    out = s.calibrate("I had toast this morning", {"pitch": 180.0, "hr": 70.0})
    assert out["samples"] == 1
    assert s.utterances[-1].raw["hr"] == 97.0 and s.utterances[-1].raw["rr"] == 19.0
