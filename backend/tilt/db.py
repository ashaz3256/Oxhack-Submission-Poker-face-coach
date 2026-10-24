"""Persistence. SQLite locally; Postgres on Tiger Data in production (see schema.sql for the hypertable)."""
from __future__ import annotations

import os
from dataclasses import asdict

from sqlalchemy import JSON, Boolean, Column, Float, Integer, MetaData, String, Table, create_engine, insert
from sqlalchemy.pool import StaticPool

from .mafia import MafiaSession

metadata = MetaData()

sessions = Table(
    "sessions", metadata,
    Column("id", String, primary_key=True),
    Column("player", String),
    Column("created_at", Float),
)

utterances = Table(
    "utterances", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("session_id", String, index=True),
    Column("t", Float, index=True),
    Column("game", Integer),
    Column("day", Integer),
    Column("kind", String),
    Column("question", String),
    Column("transcript", String),
    Column("role", String),
    Column("raw", JSON),
    Column("features", JSON),
    Column("is_lie", Boolean),
    Column("p_lie", Float),
    Column("p_blind", Float),
    Column("cues", JSON),
)


class Store:
    """Writes each session's new answers as they happen. Live game state stays in memory."""

    def __init__(self, url: str | None = None):
        url = url or os.environ.get("DATABASE_URL") or "sqlite:///tilt.db"
        kwargs = {}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
            if url in ("sqlite://", "sqlite:///:memory:"):
                kwargs["poolclass"] = StaticPool
        self.engine = create_engine(url, **kwargs)
        metadata.create_all(self.engine)
        self._written: dict[str, int] = {}

    def save(self, s: MafiaSession) -> None:
        with self.engine.begin() as conn:
            if s.id not in self._written:
                conn.execute(insert(sessions).values(id=s.id, player=s.player, created_at=s.clock()))
                self._written[s.id] = 0
            new = s.utterances[self._written[s.id]:]
            if new:
                conn.execute(insert(utterances), [{"session_id": s.id, **asdict(u)} for u in new])
                self._written[s.id] = len(s.utterances)
