"""The Mafia leak report: how well the Inspector reads your lies, and what it cost you."""
from __future__ import annotations

import numpy as np

from .mafia import MafiaSession
from .models import safe_auc

NAMES = {"hr_z": "heart rate", "hr_slope": "rising heart rate", "br_z": "breathing", "decision_s": "how long you take to answer"}


def mafia_report(s: MafiaSession) -> dict:
    labelled = [a for a in s.answers if a.is_lie is not None]
    y = [int(a.is_lie) for a in labelled]
    # Prequential: the read was made before the Inspector learned whether that answer was a lie.
    auc = safe_auc(y, [a.p_lie_reader for a in labelled])
    lies = [a for a in labelled if a.is_lie]
    truths = [a for a in labelled if not a.is_lie]

    def mean(rows, key):
        vals = [r.features.get(key) for r in rows if r.features.get(key) == r.features.get(key)]
        return round(float(np.mean(vals)), 2) if vals else None

    as_mafia = [v for v in s.votes if v.you_mafia]
    extra = [v.votes_on_you_reader - v.votes_on_you_blind for v in as_mafia]
    finished = [g for g in s.games if g.phase == "over"]
    mafia_games = [g for g in finished if g.you_mafia]

    curve = [{"answer": i, "p_lie": a.p_lie_reader, "lie": bool(a.is_lie)} for i, a in enumerate(labelled, start=1)]

    return {
        "session_id": s.id,
        "inspector": s.inspector,
        "vitals_source": s.vitals_source,
        "games": len(finished),
        "games_as_mafia": len(mafia_games),
        "mafia_wins": sum(g.winner == "mafia" for g in mafia_games),
        "answers": len(labelled),
        "lies": len(lies),
        "lie_auc": auc,
        "coefficients": s.lie_model.coefficients(),
        "means": {f: {"lie": mean(lies, f), "truth": mean(truths, f)} for f in ("hr", "hr_z", "hr_slope", "br_z", "decision_s")},
        "days_as_mafia": len(as_mafia),
        "votes_on_you_reader": round(float(np.mean([v.votes_on_you_reader for v in as_mafia])), 2) if as_mafia else None,
        "votes_on_you_blind": round(float(np.mean([v.votes_on_you_blind for v in as_mafia])), 2) if as_mafia else None,
        "extra_votes_from_tells": int(sum(extra)),
        "caught_by_reader": sum(v.out_reader for v in as_mafia),
        "caught_by_blind": sum(v.out_blind for v in as_mafia),
        "curve": curve,
    }


def mafia_headline(r: dict) -> str:
    if r["answers"] < 6:
        return "Answer a few more of the Inspector's questions so it can study you."
    if not r["days_as_mafia"]:
        return f"The Inspector has read {r['answers']} of your answers. Play a game as the mafia to see what it catches."
    coef = r["coefficients"]
    worst = max(coef.items(), key=lambda kv: abs(kv[1]), default=(None, 0))[0]
    tell = NAMES.get(worst, "your vitals")
    caught, blind = r["caught_by_reader"], r["caught_by_blind"]
    if caught <= blind:
        return f"Poker face holding: reading your vitals didn't get you caught any more often ({caught} vs {blind} days)."
    return (f"Your body gave you away: as the mafia you were voted out on {caught} of {r['days_as_mafia']} days "
            f"by an Inspector reading your vitals, versus {blind} without them. Biggest tell: {tell}.")
