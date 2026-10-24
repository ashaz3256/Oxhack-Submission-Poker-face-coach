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
from .report import headline, leak_report  # noqa: E402
from .synthetic import play  # noqa: E402

app = FastAPI(title="Tilt")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SESSIONS: dict[str, Session] = {}
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
