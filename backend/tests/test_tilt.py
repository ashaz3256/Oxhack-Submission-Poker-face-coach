import os

os.environ["DATABASE_URL"] = "sqlite://"

from fastapi.testclient import TestClient  # noqa: E402

from tilt import cards  # noqa: E402
from tilt.api import app  # noqa: E402
from tilt.game import Session  # noqa: E402
from tilt.report import leak_report  # noqa: E402
from tilt.synthetic import FakeClock, play  # noqa: E402


def test_equity_sanity():
    aces = cards.from_str(["As", "Ah"])
    junk = cards.from_str(["7c", "2d"])
    assert cards.equity_vs_random(aces, []) > 0.8
    assert cards.equity_vs_random(junk, []) < 0.4


def test_chips_are_conserved_per_hand():
    clock = FakeClock()
    s = Session(seed=3, clock=clock)
    for _ in range(15):
        s.new_hand()
        while s.current_hand.phase != "done":
            clock.advance(2)
            s.act(s.legal_actions()[0] if s.current_hand.number % 2 else s.legal_actions()[-1])
        h = s.current_hand
        assert abs(h.result_bb) <= max(h.hero_in, h.bot_in) + 1e-9
    assert round(sum(h.result_bb for h in s.hands), 6) == round(s.net_bb, 6)


def test_illegal_action_rejected():
    s = Session(seed=1)
    s.new_hand()
    try:
        s.act("call")
    except ValueError:
        return
    raise AssertionError("call should be illegal when not facing a bet")


def test_leaky_player_is_read_and_pays():
    report = leak_report(play(n_hands=40, leakiness=1.5, seed=11))
    tells = report["tells"]
    assert tells["bets"] >= 20
    assert tells["shark_auc"] is not None and tells["shark_auc"] > 0.7
    assert tells["cost_bb"] > 0


def test_api_round_trip():
    client = TestClient(app)
    sid = client.post("/api/sessions", json={"opponent": "shark"}).json()["session_id"]
    for _ in range(5):
        assert client.post(f"/api/sessions/{sid}/sim-tick").status_code == 200
    state = client.post(f"/api/sessions/{sid}/hands").json()
    assert state["hand"]["legal"] == ["check", "bet_half", "bet_pot"]
    while state["hand"]["phase"] != "done":
        state = client.post(f"/api/sessions/{sid}/action", json={"action": state["hand"]["legal"][-1]}).json()
    assert state["hand"]["bot"] is not None
    report = client.get(f"/api/sessions/{sid}/report").json()
    assert report["hands"] == 1 and "headline" in report
    assert client.post(f"/api/sessions/{sid}/review").json()["text"]
    assert client.post("/api/speak", json={"text": "hi"}).status_code in (200, 204)


def test_mafia_liar_is_caught_more_by_the_reader():
    from tilt.mafia_report import mafia_report
    from tilt.synthetic import play_mafia

    r = mafia_report(play_mafia(n_games=20, leakiness=1.5, seed=2))
    assert r["lie_auc"] > 0.7
    assert r["caught_by_reader"] > r["caught_by_blind"]


def test_mafia_api_round_trip():
    client = TestClient(app)
    sid = client.post("/api/mafia/sessions", json={"inspector": "reader"}).json()["session_id"]
    for _ in range(3):
        assert client.post(f"/api/mafia/sessions/{sid}/sim-tick").status_code == 200
    state = client.post(f"/api/mafia/sessions/{sid}/games").json()
    while state["game"]["phase"] != "over":
        g = state["game"]
        if g["phase"] == "question":
            state = client.post(f"/api/mafia/sessions/{sid}/answer", json={"option": g["question"]["options"][0]["id"]}).json()
        elif g["phase"] == "vote":
            state = client.post(f"/api/mafia/sessions/{sid}/vote", json={"target": g["vote_options"][0]}).json()
        else:
            state = client.post(f"/api/mafia/sessions/{sid}/kill", json={"target": g["vote_options"][0]}).json()
    report = client.get(f"/api/mafia/sessions/{sid}/report").json()
    assert report["games"] == 1 and report["headline"]
    assert client.post(f"/api/mafia/sessions/{sid}/review").json()["text"]
