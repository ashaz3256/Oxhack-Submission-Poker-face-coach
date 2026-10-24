"""The leak report: what your body gave away, and what it cost."""
from __future__ import annotations

import numpy as np

from .game import Session
from .models import safe_auc

BET_ACTIONS = ("bet_half", "bet_pot")


def leak_report(session: Session) -> dict:
    hands = [h for h in session.hands if h.result_bb is not None]
    bets = [d for d in session.decisions if d.action in BET_ACTIONS]
    facing = [d for d in session.decisions if d.is_bad is not None]

    # Prequential: the Shark's read *at the time of each bet*, before it learned that bet's label.
    y = [int(bool(d.is_bluff)) for d in bets]
    shark_auc = safe_auc(y, [d.p_bluff_shark for d in bets])

    cost = [d.ev_vs_honest - d.ev_vs_shark for d in bets]
    cost_total = float(np.sum(cost)) if cost else 0.0
    # Standard error of the total, so a small sample is not oversold.
    cost_se = float(np.std(cost, ddof=1) * np.sqrt(len(cost))) if len(cost) > 1 else None
    n_hands = max(len(hands), 1)

    # Learning curve: cumulative chips lost to tells, bet by bet.
    curve = []
    running = 0.0
    for i, (d, c) in enumerate(zip(bets, cost), start=1):
        running += c
        curve.append({"bet": i, "hand": d.hand_number, "cost_bb": round(running, 2),
                      "p_bluff": d.p_bluff_shark, "bluff": bool(d.is_bluff)})

    bluffs = [d for d in bets if d.is_bluff]
    values = [d for d in bets if not d.is_bluff]

    def mean(rows, key):
        vals = [r.features.get(key) for r in rows if r.features.get(key) == r.features.get(key)]
        return round(float(np.mean(vals)), 2) if vals else None

    tells = {
        f: {"bluff": mean(bluffs, f), "value": mean(values, f)}
        for f in ("hr", "hr_z", "hr_slope", "br_z", "decision_s")
    }

    stressed = [d for d in facing if d.features.get("hr_z", 0) > 1]
    calm = [d for d in facing if d.features.get("hr_z", 0) <= 1]

    def rate(rows):
        return round(sum(bool(d.is_bad) for d in rows) / len(rows), 3) if rows else None

    return {
        "session_id": session.id,
        "opponent": session.opponent,
        "vitals_source": session.vitals_source,
        "hands": len(hands),
        "net_bb": round(session.net_bb, 2),
        "bb_per_100": round(100 * session.net_bb / n_hands, 1),
        "tells": {
            "bets": len(bets),
            "bluffs": len(bluffs),
            "shark_auc": shark_auc,
            "coefficients": session.tell_model.coefficients(),
            "means": tells,
            "cost_bb": round(cost_total, 2),
            "cost_se_bb": None if cost_se is None else round(cost_se, 2),
            "cost_bb_per_100": round(100 * cost_total / n_hands, 1),
            "curve": curve,
        },
        "tilt": {
            "decisions": len(facing),
            "bad": sum(bool(d.is_bad) for d in facing),
            "bad_rate_stressed": rate(stressed),
            "bad_rate_calm": rate(calm),
            "model_auc": safe_auc([int(bool(d.is_bad)) for d in facing], [d.p_bad for d in facing]),
            "coefficients": session.tilt_model.coefficients(),
        },
    }


def headline(report: dict) -> str:
    t = report["tells"]
    if t["bets"] < 5:
        return "Play a few more hands so the Shark can study you."
    if t["cost_bb"] <= 0.5:
        return f"Poker face holding: the Shark has read {t['bets']} of your bets and gained nothing."
    worst = max(t["coefficients"].items(), key=lambda kv: abs(kv[1]), default=(None, 0))[0]
    names = {"hr_z": "heart rate", "hr_slope": "rising heart rate", "br_z": "breathing",
             "decision_s": "how long you think"}
    tell = names.get(worst, "your vitals")
    return (f"Your body leaked {t['cost_bb']:g} big blinds over {report['hands']} hands "
            f"({t['cost_bb_per_100']:g} bb/100). Biggest tell: {tell}.")
