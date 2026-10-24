"""How well does the AI town reason? Play many simulated games through the real engine and measure.

Uses the simulated player in backend/tilt/synthetic.py, whose face and voice shift when lying by
`leakiness` (0 = perfect poker face). These are SIMULATED players: the numbers say whether the
town's reasoning is sound and calibrated, not how well it reads real people.

    python research/simulate_town.py                       # default sweep
    python research/simulate_town.py --sessions 30 --games 10 --leak 0 0.5 1 1.5

Measures, per leakiness level (95% intervals from a bootstrap over sessions):
- caught rate: share of days as the mafia on which the human was voted out, Reader vs Blind town
  (same dice; the Blind town never hears the lie detector), and the paired difference
- false accusations: share of days as a villager on which the human was wrongly voted out
- calibration: Brier score and a reliability table of every villager bot's P(player is mafia) at
  every vote, against who the mafia really was
Writes research/results/town_eval_game_deal.json by default.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from tilt import synthetic  # noqa: E402


def session_stats(s) -> dict:
    mafia_days = [v for v in s.votes if v.you_mafia]
    villager_days = [v for v in s.votes if not v.you_mafia]
    finished = [g for g in s.games if g.phase == "over"]
    probs, truth, probs_b = [], [], []
    for row in s.belief_log:
        for bot, post in row["beliefs"].items():
            blind = row["beliefs_blind"][bot]
            for cand, p in post.items():
                probs.append(p)
                probs_b.append(blind[cand])
                truth.append(int(cand == row["mafia"]))
    return {
        "mafia_days": len(mafia_days),
        "caught_reader": sum(v.out_reader for v in mafia_days),
        "caught_blind": sum(v.out_blind for v in mafia_days),
        "villager_days": len(villager_days),
        "framed_reader": sum(v.out_reader for v in villager_days),
        "framed_blind": sum(v.out_blind for v in villager_days),
        "games": len(finished),
        "town_wins": sum(g.winner == "town" for g in finished),
        "human_wins": sum((g.winner == "mafia") == g.you_mafia for g in finished),
        "probs": probs, "probs_blind": probs_b, "truth": truth,
    }


def ratio(rows, num, den):
    n, d = sum(r[num] for r in rows), sum(r[den] for r in rows)
    return n / d if d else float("nan")


def bootstrap(rows, fn, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        sample = [rows[i] for i in rng.integers(0, len(rows), len(rows))]
        v = fn(sample)
        if v == v:
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def brier(p, y):
    p, y = np.asarray(p), np.asarray(y)
    return float(np.mean((p - y) ** 2))


def reliability(p, y, bins=(0, .1, .2, .35, .5, .7, 1.0001)):
    p, y = np.asarray(p), np.asarray(y)
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum():
            out.append({"bin": f"{lo:.2f}-{min(hi, 1):.2f}", "mean_p": round(float(p[m].mean()), 3),
                        "actual_rate": round(float(y[m].mean()), 3), "n": int(m.sum())})
    return out


def evaluate(leak: float, sessions: int, games: int, seed0: int = 100, mafia_rate: float = 0.5,
             same_words: bool = False) -> dict:
    rows = []
    for i in range(sessions):
        with contextlib.redirect_stdout(io.StringIO()):
            s = synthetic.play(n_games=games, leakiness=leak, seed=seed0 + i, mafia_rate=mafia_rate,
                               same_words=same_words)
        rows.append(session_stats(s))
    probs = [p for r in rows for p in r["probs"]]
    probs_b = [p for r in rows for p in r["probs_blind"]]
    truth = [t for r in rows for t in r["truth"]]

    def diff(rs):
        return ratio(rs, "caught_reader", "mafia_days") - ratio(rs, "caught_blind", "mafia_days")

    def brier_of(key):
        return lambda rs: brier([p for r in rs for p in r[key]], [t for r in rs for t in r["truth"]])

    return {
        "leakiness": leak, "mafia_rate": mafia_rate, "same_words": same_words, "sessions": sessions, "games": sum(r["games"] for r in rows),
        "mafia_days": sum(r["mafia_days"] for r in rows), "villager_days": sum(r["villager_days"] for r in rows),
        "caught_reader": ratio(rows, "caught_reader", "mafia_days"),
        "caught_reader_95ci": bootstrap(rows, lambda rs: ratio(rs, "caught_reader", "mafia_days")),
        "caught_blind": ratio(rows, "caught_blind", "mafia_days"),
        "caught_blind_95ci": bootstrap(rows, lambda rs: ratio(rs, "caught_blind", "mafia_days")),
        "caught_difference": diff(rows), "caught_difference_95ci": bootstrap(rows, diff),
        "framed_reader": ratio(rows, "framed_reader", "villager_days"),
        "framed_blind": ratio(rows, "framed_blind", "villager_days"),
        "framed_reader_95ci": bootstrap(rows, lambda rs: ratio(rs, "framed_reader", "villager_days")),
        "town_win_rate": ratio(rows, "town_wins", "games"),
        "human_win_rate": ratio(rows, "human_wins", "games"),
        "brier_reader": brier(probs, truth), "brier_reader_95ci": bootstrap(rows, brier_of("probs")),
        "brier_blind": brier(probs_b, truth),
        "brier_uniform": brier([1 / max(1, n) for n in _cands_per_row(probs, truth)], truth),
        "reliability_reader": reliability(probs, truth),
    }


def _cands_per_row(probs, truth):
    # Uniform baseline: each posterior has k candidates and exactly one is the mafia, so p = 1/k.
    # Recover k from the data: probabilities of one posterior sum to 1 and contain one positive.
    out, k, acc = [], 0, 0.0
    for p, _t in zip(probs, truth):
        k += 1
        acc += p
        if acc > 0.999:
            out.extend([k] * k)
            k, acc = 0, 0.0
    out.extend([max(k, 1)] * k)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leak", type=float, nargs="+", default=[0.0, 0.5, 1.0, 1.5])
    ap.add_argument("--sessions", type=int, default=20)
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--mafia-rate", type=float, default=0.5, help="how often the human is dealt the mafia")
    ap.add_argument("--same-words", action="store_true", help="placebo: liars say the same things as truth-tellers")
    ap.add_argument("--out", type=Path, default=ROOT / "research" / "results" / "town_eval_game_deal.json")
    a = ap.parse_args()
    results = []
    for leak in a.leak:
        r = evaluate(leak, a.sessions, a.games, mafia_rate=a.mafia_rate, same_words=a.same_words)
        results.append(r)
        print(f"leak {leak:>3}: caught as mafia {r['caught_reader']:.2f} reader vs {r['caught_blind']:.2f} blind "
              f"(diff {r['caught_difference']:+.2f}, 95% CI {r['caught_difference_95ci'][0]:+.2f}..{r['caught_difference_95ci'][1]:+.2f}); "
              f"wrongly voted out as villager {r['framed_reader']:.2f} vs {r['framed_blind']:.2f}; "
              f"Brier {r['brier_reader']:.3f} (blind {r['brier_blind']:.3f}, uniform {r['brier_uniform']:.3f}); "
              f"{r['mafia_days']} mafia days")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"note": "simulated players (backend/tilt/synthetic.py), not real people",
                                 "settings": {"mafia_rate": a.mafia_rate, "same_words": a.same_words},
                                 "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
