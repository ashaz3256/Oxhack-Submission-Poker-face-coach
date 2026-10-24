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
    client.post(f"/api/sessions/{sid}/calibrate", json={"transcript": "hi I had a good day", "raw": {"pitch": 140}})
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
