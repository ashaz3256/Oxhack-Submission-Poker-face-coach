"""HTTP API for the spoken Mafia call."""
from __future__ import annotations

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

load_dotenv()

from . import coach  # noqa: E402
from .db import Store  # noqa: E402
from .mafia import MafiaSession  # noqa: E402
from .report import headline, report  # noqa: E402
from .synthetic import play  # noqa: E402

app = FastAPI(title="Tilt")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SESSIONS: dict[str, MafiaSession] = {}
store = Store()


class NewSession(BaseModel):
    player: str = "player"
    mafia_rate: float = Field(0.5, ge=0, le=1)


class Said(BaseModel):
    transcript: str = ""
    raw: dict[str, float | None] = Field(default_factory=dict)
    # Per-frame face primitives for this turn (frontend/src/sensors/faceFrame.js), for the dataset-trained model.
    face_frames: list[list[float | None]] | None = Field(None, max_length=6000)


class Speech(BaseModel):
    text: str = Field(max_length=500)
    speaker: str = "Narrator"


def _session(sid: str) -> MafiaSession:
    if sid not in SESSIONS:
        raise HTTPException(404, "unknown session")
    return SESSIONS[sid]


def _step(s: MafiaSession, fn) -> dict:
    try:
        fn()
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    store.save(s)
    return s.view()


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.post("/api/sessions")
def create(body: NewSession) -> dict:
    s = MafiaSession(player=body.player, mafia_rate=body.mafia_rate)
    SESSIONS[s.id] = s
    store.save(s)
    return {"session_id": s.id}


@app.post("/api/sessions/{sid}/calibrate")
def calibrate(sid: str, body: Said) -> dict:
    s = _session(sid)
    s.face_frames = body.face_frames
    out = s.calibrate(body.transcript, body.raw)
    store.save(s)
    return out


@app.post("/api/sessions/{sid}/games")
def new_game(sid: str) -> dict:
    s = _session(sid)
    return _step(s, s.new_game)


@app.post("/api/sessions/{sid}/speak")
def speak(sid: str, body: Said) -> dict:
    s = _session(sid)
    def turn():
        s.face_frames = body.face_frames
        try:
            s.speak(body.transcript, body.raw)
        finally:
            s.face_frames = None  # frames belong to this turn only, used or not
    return _step(s, turn)


@app.get("/api/face-model")
def face_model_status() -> dict:
    """Whether a dataset-trained face model is installed, and how it scored on held-out participants."""
    from .face_model import FaceScorer
    return FaceScorer().status()


@app.get("/api/sessions/{sid}/report")
def get_report(sid: str) -> dict:
    r = report(_session(sid))
    return {**r, "headline": headline(r)}


@app.post("/api/sessions/{sid}/review")
def get_review(sid: str) -> dict:
    r = report(_session(sid))
    return coach.review(r, headline(r))


@app.post("/api/demo")
def demo(games: int = 15, leakiness: float = 1.0) -> dict:
    """A finished simulated session, so the report has something to show if the live demo fails."""
    s = play(n_games=min(games, 40), leakiness=leakiness)
    SESSIONS[s.id] = s
    store.save(s)
    return {"session_id": s.id}


@app.post("/api/tts")
def tts(body: Speech) -> Response:
    audio = coach.speak(body.text, body.speaker)
    if audio is None:
        return Response(status_code=204)
    return Response(content=audio, media_type="audio/mpeg")
