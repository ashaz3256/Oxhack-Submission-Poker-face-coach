"""HTTP API for the table, the vitals stream and the leak report."""
from __future__ import annotations

from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

load_dotenv()

from . import coach  # noqa: E402
from .db import Store  # noqa: E402
from .game import Session  # noqa: E402
from .mafia import MafiaSession  # noqa: E402
from .mafia_report import mafia_headline, mafia_report  # noqa: E402
from .report import headline, leak_report  # noqa: E402
from .synthetic import play, play_mafia  # noqa: E402

app = FastAPI(title="Tilt")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SESSIONS: dict[str, Session] = {}
MAFIA: dict[str, MafiaSession] = {}
store = Store()


class NewSession(BaseModel):
    opponent: Literal["shark", "honest"] = "shark"
    vitals_source: Literal["sim", "webcam", "presage"] = "sim"
    player: str = "player"
    leakiness: float = Field(1.0, ge=0, le=3)


class Action(BaseModel):
    action: Literal["check", "bet_half", "bet_pot", "call", "fold"]


class VitalsSample(BaseModel):
    t: float
    hr: float
    br: float | None = None


class VitalsBatch(BaseModel):
    samples: list[VitalsSample]


class Speech(BaseModel):
    text: str = Field(max_length=400)


def _session(sid: str) -> Session:
    if sid not in SESSIONS:
        raise HTTPException(404, "unknown session")
    return SESSIONS[sid]


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.post("/api/sessions")
def create_session(body: NewSession) -> dict:
    s = Session(opponent=body.opponent, vitals_source=body.vitals_source, player=body.player, leakiness=body.leakiness)
    SESSIONS[s.id] = s
    store.save(s)
    return {"session_id": s.id, **s.hand_view()}


@app.post("/api/sessions/{sid}/hands")
def new_hand(sid: str) -> dict:
    s = _session(sid)
    s.new_hand()
    return s.hand_view()


@app.post("/api/sessions/{sid}/action")
def act(sid: str, body: Action) -> dict:
    s = _session(sid)
    try:
        s.act(body.action)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    store.save(s)
    return {**s.hand_view(), "live": coach.live_read(s)}


@app.post("/api/sessions/{sid}/vitals")
def add_vitals(sid: str, body: VitalsBatch) -> dict:
    s = _session(sid)
    s.add_samples([v.model_dump() for v in body.samples])
    store.save(s)
    return {"stored": len(body.samples)}


@app.post("/api/sessions/{sid}/sim-tick")
def sim_tick(sid: str) -> dict:
    s = _session(sid)
    try:
        sample = s.sim_tick()
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return sample


@app.get("/api/sessions/{sid}/live")
def live(sid: str) -> dict:
    return coach.live_read(_session(sid))


@app.get("/api/sessions/{sid}/report")
def report(sid: str) -> dict:
    r = leak_report(_session(sid))
    return {**r, "headline": headline(r)}


@app.post("/api/sessions/{sid}/review")
def review(sid: str) -> dict:
    return coach.session_review(_session(sid))


@app.post("/api/demo")
def demo(hands: int = 60, leakiness: float = 1.0) -> dict:
    """A finished simulated session, so the report page has something to show on stage."""
    s = play(n_hands=min(hands, 150), leakiness=leakiness)
    SESSIONS[s.id] = s
    store.save(s)
    return {"session_id": s.id}


@app.post("/api/speak")
def speak(body: Speech) -> Response:
    audio = coach.speak(body.text)
    if audio is None:
        return Response(status_code=204)
    return Response(content=audio, media_type="audio/mpeg")


# ---- Mafia ---------------------------------------------------------------------------------------
class NewMafia(BaseModel):
    inspector: Literal["reader", "blind"] = "reader"
    vitals_source: Literal["sim", "webcam", "presage"] = "sim"
    player: str = "player"
    leakiness: float = Field(1.0, ge=0, le=3)


class Choice(BaseModel):
    option: str | None = None
    target: str | None = None


def _mafia(sid: str) -> MafiaSession:
    if sid not in MAFIA:
        raise HTTPException(404, "unknown session")
    return MAFIA[sid]


def _mafia_live(s: MafiaSession) -> dict:
    from .features import TELL_FEATURES, vector, vitals_features

    now = s.clock()
    f = vitals_features(s.samples, now)
    game = s.current
    q = game.question if game else None
    f["decision_s"] = now - q.asked_at if q else 0.0
    p = s.lie_model.predict(vector(f, TELL_FEATURES))
    latest = s.samples[-1] if s.samples else None
    if not s.lie_model.ready:
        line = "The Inspector is still studying you."
    elif q and p > 0.6:
        line = f"Careful. The Inspector reads {p:.0%} that your next answer is a lie."
    elif f["hr_z"] > 1.5:
        line = "Your heart rate is up. Breathe out slowly before you answer."
    else:
        line = "You look calm."
    return {"hr": None if latest is None else latest.hr, "br": None if latest is None else latest.br,
            "hr_z": round(f["hr_z"], 2), "p_lie": round(p, 3), "ready": s.lie_model.ready, "line": line}


def _mafia_step(s: MafiaSession, fn) -> dict:
    try:
        fn()
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    store.save_mafia(s)
    return {**s.view(), "live": _mafia_live(s)}


@app.post("/api/mafia/sessions")
def mafia_create(body: NewMafia) -> dict:
    s = MafiaSession(inspector=body.inspector, vitals_source=body.vitals_source, player=body.player,
                     leakiness=body.leakiness)
    MAFIA[s.id] = s
    store.save_mafia(s)
    return {"session_id": s.id}


@app.post("/api/mafia/sessions/{sid}/games")
def mafia_new_game(sid: str) -> dict:
    s = _mafia(sid)
    return _mafia_step(s, s.new_game)


@app.post("/api/mafia/sessions/{sid}/answer")
def mafia_answer(sid: str, body: Choice) -> dict:
    s = _mafia(sid)
    return _mafia_step(s, lambda: s.answer(body.option or ""))


@app.post("/api/mafia/sessions/{sid}/vote")
def mafia_vote(sid: str, body: Choice) -> dict:
    s = _mafia(sid)
    return _mafia_step(s, lambda: s.vote(body.target or ""))


@app.post("/api/mafia/sessions/{sid}/kill")
def mafia_kill(sid: str, body: Choice) -> dict:
    s = _mafia(sid)
    return _mafia_step(s, lambda: s.kill(body.target or ""))


@app.post("/api/mafia/sessions/{sid}/vitals")
def mafia_vitals(sid: str, body: VitalsBatch) -> dict:
    s = _mafia(sid)
    s.add_samples([v.model_dump() for v in body.samples])
    store.save_mafia(s)
    return {"stored": len(body.samples)}


@app.post("/api/mafia/sessions/{sid}/sim-tick")
def mafia_sim_tick(sid: str) -> dict:
    s = _mafia(sid)
    try:
        return s.sim_tick()
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/mafia/sessions/{sid}/live")
def mafia_live(sid: str) -> dict:
    return _mafia_live(_mafia(sid))


@app.get("/api/mafia/sessions/{sid}/report")
def mafia_get_report(sid: str) -> dict:
    r = mafia_report(_mafia(sid))
    return {**r, "headline": mafia_headline(r)}


@app.post("/api/mafia/sessions/{sid}/review")
def mafia_review(sid: str) -> dict:
    r = mafia_report(_mafia(sid))
    r.pop("curve", None)
    return coach.review_from_report(r, mafia_headline(r), game="Mafia")


@app.post("/api/mafia/demo")
def mafia_demo(games: int = 15, leakiness: float = 1.0) -> dict:
    s = play_mafia(n_games=min(games, 40), leakiness=leakiness)
    MAFIA[s.id] = s
    store.save_mafia(s)
    return {"session_id": s.id}
