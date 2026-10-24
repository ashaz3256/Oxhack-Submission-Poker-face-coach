"""The leak report: how well your face, voice and words gave your lies away, and what it cost you."""
from __future__ import annotations

import numpy as np

from .cues import LABELS, RAW
from .mafia import MafiaSession
from .models import safe_auc


def report(s: MafiaSession) -> dict:
    answers = [u for u in s.utterances if u.kind == "answer" and u.is_lie is not None]
    lies = [u for u in answers if u.is_lie]
    truths = [u for u in answers if not u.is_lie]
    # Prequential: each read was made before the model learned whether that answer was a lie.
    auc = safe_auc([int(u.is_lie) for u in answers], [u.p_lie for u in answers])

    def mean(rows, key, raw=True):
        vals = [(r.raw if raw else r.features).get(key) for r in rows]
        vals = [v for v in vals if v is not None and v == v]
        return round(float(np.mean(vals)), 3) if vals else None

    cues = []
    for k in RAW:
        cues.append({"key": k, "label": LABELS[f"{k}_z"], "lie": mean(lies, k), "truth": mean(truths, k),
                     "lie_z": mean(lies, f"{k}_z", raw=False), "truth_z": mean(truths, f"{k}_z", raw=False)})
    for k in ("hedges", "fillers", "negations"):
        cues.append({"key": k, "label": LABELS[k], "lie": mean(lies, k, raw=False),
                     "truth": mean(truths, k, raw=False), "lie_z": None, "truth_z": None})

    as_mafia = [v for v in s.votes if v.you_mafia]
    finished = [g for g in s.games if g.phase == "over"]
    mafia_games = [g for g in finished if g.you_mafia]
    coef = {LABELS.get(k, k): v for k, v in s.lie_model.coefficients().items()}
    return {
        "session_id": s.id,
        "games": len(finished),
        "games_as_mafia": len(mafia_games),
        "mafia_wins": sum(g.winner == "mafia" for g in mafia_games),
        "answers": len(answers),
        "lies": len(lies),
        "lie_auc": auc,
        "coefficients": coef,
        "cues": cues,
        "days_as_mafia": len(as_mafia),
        "votes_on_you_reader": round(float(np.mean([v.votes_on_you_reader for v in as_mafia])), 2) if as_mafia else None,
        "votes_on_you_blind": round(float(np.mean([v.votes_on_you_blind for v in as_mafia])), 2) if as_mafia else None,
        "caught_by_reader": sum(v.out_reader for v in as_mafia),
        "caught_by_blind": sum(v.out_blind for v in as_mafia),
        "timeline": [{"i": i, "p_lie": u.p_lie, "lie": bool(u.is_lie), "text": u.transcript, "cues": u.cues,
                      "game": u.game, "day": u.day} for i, u in enumerate(answers, start=1)],
    }


def headline(r: dict) -> str:
    if r["answers"] < 4:
        return "Answer a few more questions so the lie detector can learn you."
    if not r["days_as_mafia"]:
        return f"The detector has studied {r['answers']} of your answers. Play a game as the mafia to see what it catches."
    worst = max(r["coefficients"].items(), key=lambda kv: kv[1], default=(None, 0))[0] or "your face and voice"
    caught, blind = r["caught_by_reader"], r["caught_by_blind"]
    if caught <= blind:
        return f"Poker face holding: your tells didn't get you caught any more often ({caught} vs {blind} days)."
    return (f"Your body gave you away: as the mafia you were voted out on {caught} of {r['days_as_mafia']} days, "
            f"versus {blind} if nobody could read you. Biggest tell: {worst}.")
