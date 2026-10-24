"""HTTP API for the spoken Mafia call."""
from __future__ import annotations

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

load_dotenv()

from . import coach, presage  # noqa: E402
from .breakdown import breakdown  # noqa: E402
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
    is_lie: bool = False      # practice round only: were you told to lie?
    question: str = ""


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
    import os

    return {"ok": True, "elevenlabs": bool(os.environ.get("ELEVENLABS_API_KEY")),
            "gemini": bool(os.environ.get("GEMINI_API_KEY")), "presage": bool(presage.api_key())}


@app.post("/api/sessions")
def create(body: NewSession) -> dict:
    s = MafiaSession(player=body.player, mafia_rate=body.mafia_rate)
    SESSIONS[s.id] = s
    store.save(s)
    return {"session_id": s.id}


@app.post("/api/sessions/{sid}/calibrate")
def calibrate(sid: str, body: Said) -> dict:
    s = _session(sid)
    out = s.calibrate(body.transcript, body.raw, is_lie=body.is_lie, question=body.question)
    store.save(s)
    return out


@app.post("/api/sessions/{sid}/games")
def new_game(sid: str) -> dict:
    s = _session(sid)
    return _step(s, s.new_game)


@app.post("/api/sessions/{sid}/speak")
def speak(sid: str, body: Said) -> dict:
    s = _session(sid)
    return _step(s, lambda: s.speak(body.transcript, body.raw))


@app.get("/api/sessions/{sid}/report")
def get_report(sid: str) -> dict:
    r = report(_session(sid))
    return {**r, "headline": headline(r)}


@app.get("/api/sessions/{sid}/breakdown")
def get_breakdown(sid: str, game: int | None = None) -> dict:
    return breakdown(_session(sid), game)


MAX_CLIP = 40 * 1024 * 1024


@app.post("/api/sessions/{sid}/vitals")
async def upload_vitals(sid: str, request: Request, ext: str = "webm") -> dict:
    """A ~15 second webcam clip for Presage. Returns at once; the reading arrives in the background."""
    s = _session(sid)
    data = await request.body()
    if not data or len(data) > MAX_CLIP:
        raise HTTPException(413, "clip missing or too large")
    ext = ext.lower() if ext.lower() in ("webm", "mp4", "mov", "avi", "mkv") else "webm"
    queued = s.presage.submit(data, ext)
    return {"queued": queued, **s.presage.view()}


@app.get("/api/sessions/{sid}/vitals")
def get_vitals(sid: str) -> dict:
    return _session(sid).presage.view()


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
