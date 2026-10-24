"""Persistence. SQLite locally; Postgres on Tiger Data in production (see schema.sql for the hypertable)."""
from __future__ import annotations

import os
from dataclasses import asdict

from sqlalchemy import (JSON, Boolean, Column, Float, Integer, MetaData, String, Table, create_engine, insert,
                        update)
from sqlalchemy.pool import StaticPool

from . import cards
from .game import Session

metadata = MetaData()

sessions = Table(
    "sessions", metadata,
    Column("id", String, primary_key=True),
    Column("player", String),
    Column("opponent", String),
    Column("vitals_source", String),
    Column("created_at", Float),
    Column("net_bb", Float),
)

vitals = Table(
    "vitals", metadata,
    Column("session_id", String, index=True),
    Column("t", Float, index=True),
    Column("hr", Float),
    Column("br", Float),
)

hands = Table(
    "hands", metadata,
    Column("id", String, primary_key=True),
    Column("session_id", String, index=True),
    Column("number", Integer),
    Column("hero", String),
    Column("bot", String),
    Column("board", String),
    Column("started_at", Float),
    Column("result_bb", Float),
    Column("showdown", Boolean),
)

decisions = Table(
    "decisions", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("session_id", String, index=True),
    Column("hand_id", String),
    Column("hand_number", Integer),
    Column("street", String),
    Column("t", Float),
    Column("action", String),
    Column("amount", Float),
    Column("pot_before", Float),
    Column("to_call", Float),
    Column("strength", Float),
    Column("pot_odds", Float),
    Column("is_bluff", Boolean),
    Column("is_bad", Boolean),
    Column("features", JSON),
    Column("p_bluff_shark", Float),
    Column("p_bluff_honest", Float),
    Column("response_shark", String),
    Column("response_honest", String),
    Column("ev_vs_shark", Float),
    Column("ev_vs_honest", Float),
    Column("p_bad", Float),
)


answers = Table(
    "mafia_answers", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("session_id", String, index=True),
    Column("game", Integer),
    Column("day", Integer),
    Column("question", String),
    Column("option", String),
    Column("text", String),
    Column("t", Float),
    Column("is_lie", Boolean),
    Column("role", String),
    Column("features", JSON),
    Column("p_lie_reader", Float),
    Column("p_lie_blind", Float),
)


class Store:
    """Writes each session's new rows as they appear. Live game state stays in memory."""

    def __init__(self, url: str | None = None):
        url = url or os.environ.get("DATABASE_URL", "sqlite:///tilt.db")
        kwargs = {}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
            if url in ("sqlite://", "sqlite:///:memory:"):
                kwargs["poolclass"] = StaticPool
        self.engine = create_engine(url, **kwargs)
        metadata.create_all(self.engine)
        self._written: dict[str, dict[str, int]] = {}

    def save(self, s: Session) -> None:
        done = self._written.setdefault(s.id, {"created": 0, "vitals": 0, "decisions": 0, "hands": 0})
        with self.engine.begin() as conn:
            if not done["created"]:
                conn.execute(insert(sessions).values(id=s.id, player=s.player, opponent=s.opponent,
                                                     vitals_source=s.vitals_source, created_at=s.clock(), net_bb=0.0))
                done["created"] = 1
            new_vitals = s.samples[done["vitals"]:]
            if new_vitals:
                conn.execute(insert(vitals), [{"session_id": s.id, "t": v.t, "hr": v.hr, "br": v.br} for v in new_vitals])
                done["vitals"] = len(s.samples)
            new_decisions = s.decisions[done["decisions"]:]
            if new_decisions:
                conn.execute(insert(decisions), [{"session_id": s.id, **asdict(d)} for d in new_decisions])
                done["decisions"] = len(s.decisions)
            finished = [h for h in s.hands if h.result_bb is not None]
            for h in finished[done["hands"]:]:
                conn.execute(insert(hands).values(
                    id=h.id, session_id=s.id, number=h.number, hero=" ".join(cards.to_str(h.hero)),
                    bot=" ".join(cards.to_str(h.bot)), board=" ".join(cards.to_str(h.board_full)),
                    started_at=h.started_at, result_bb=h.result_bb, showdown=h.showdown))
            done["hands"] = len(finished)
            conn.execute(update(sessions).where(sessions.c.id == s.id).values(net_bb=s.net_bb))

    def save_mafia(self, s) -> None:
        done = self._written.setdefault(s.id, {"created": 0, "vitals": 0, "answers": 0})
        with self.engine.begin() as conn:
            if not done["created"]:
                conn.execute(insert(sessions).values(id=s.id, player=s.player, opponent=f"mafia-{s.inspector}",
                                                     vitals_source=s.vitals_source, created_at=s.clock(), net_bb=0.0))
                done["created"] = 1
            new_vitals = s.samples[done["vitals"]:]
            if new_vitals:
                conn.execute(insert(vitals), [{"session_id": s.id, "t": v.t, "hr": v.hr, "br": v.br} for v in new_vitals])
                done["vitals"] = len(s.samples)
            new_answers = s.answers[done["answers"]:]
            if new_answers:
                conn.execute(insert(answers), [{"session_id": s.id, **asdict(a)} for a in new_answers])
                done["answers"] = len(s.answers)
