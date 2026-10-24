"""HTTP API for the spoken Mafia call."""
from __future__ import annotations

import os

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
    return {"ok": True, "elevenlabs": bool(os.environ.get("ELEVENLABS_API_KEY")), "voice_error": coach.last_error(),
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
        reason = (coach.last_error() or "no voice").encode("ascii", "replace").decode()[:200]
        return Response(status_code=204, headers={"X-TTS-Error": reason})
    return Response(content=audio, media_type="audio/mpeg")


@app.post("/api/transcribe")
async def transcribe(request: Request) -> dict:
    """Backup speech-to-text for browsers whose built-in recognition fails: a WAV of one answer in,
    Gemini's transcript out."""
    from . import gemini

    data = await request.body()
    if not data or len(data) > 8 * 1024 * 1024:
        raise HTTPException(413, "audio missing or too long")
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(503, "backup speech-to-text needs GEMINI_API_KEY")
    text = gemini.generate(
        "Transcribe this speech exactly, in English. Reply with only the words spoken, no quotes or notes. "
        "If nobody speaks, reply with nothing.", audio=("audio/wav", data), temperature=0, deadline=15)
    if text is None:
        raise HTTPException(502, "Gemini could not transcribe that")
    return {"text": text.strip().strip('"')}


@app.get("/api/voice-check")
def voice_check() -> dict:
    """Open this in a browser to see whether ElevenLabs works with your key, and if not, why."""
    audio = coach.speak("Testing, one, two.", "Narrator")
    return {"ok": audio is not None, "bytes": len(audio or b""), "error": coach.last_error()}
