"""How much does the warm-up help the lie detector? Simulated players, same dice for every arm.

    python research/warmup_experiment.py            # writes results/warmup_experiment.json

Each session is played four times with 0, 2, 4 and 6 warm-up answers (truth, lie, truth, ...),
using the same seed, so the only difference is the warm-up. For every in-game answer we keep the
detector's read *before* it learns that answer (the game's own prequential read) and score:

- AUC on the first game's answers (where the warm-up should matter most) and on all games
- how often the human mafia is voted out

Intervals bootstrap over sessions. These are simulated players with made-up cue sizes, so the
numbers say whether the warm-up mechanism works, not how well real people can be read.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from tilt.synthetic import play  # noqa: E402

ARMS = [0, 2, 4, 6]
SESSIONS = 60
GAMES = 4
LEAKINESS = [0.5, 1.0]
# Words: "scripted" gives liars giveaway phrases (easy); "same" makes them say what truth-tellers say,
# so only face and voice can give them away, which is what the warm-up baseline is for.
WORDS = {"scripted": False, "same": True}
BOOT = 2000


def answers(s, first_game_only: bool):
    rows = [u for u in s.utterances if u.kind == "answer" and u.is_lie is not None and u.p_lie is not None]
    if first_game_only:
        rows = [u for u in rows if u.game == min((r.game for r in rows), default=0)]
    return [(u.p_lie, int(u.is_lie)) for u in rows]


def caught(s) -> list[int]:
    """1 per game where you were the mafia and the town voted you out."""
    return [int(any(v.game == g.number and v.out_reader for v in s.votes)) for g in s.games if g.you_mafia]


def pooled_auc(pairs) -> float:
    y = [l for _, l in pairs]
    return float(roc_auc_score(y, [p for p, _ in pairs])) if 0 < sum(y) < len(y) else float("nan")


def boot_ci(per_session: list, stat, rng) -> list[float]:
    vals = []
    for _ in range(BOOT):
        pick = rng.integers(0, len(per_session), len(per_session))
        v = stat([per_session[i] for i in pick])
        if v == v:
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main() -> int:
    rng = np.random.default_rng(0)
    result = {"sessions": SESSIONS, "games_per_session": GAMES, "arms": ARMS, "by_condition": {}}
    for (words, same), leak in [(w, l) for w in WORDS.items() for l in LEAKINESS]:
        runs = {a: [play(n_games=GAMES, leakiness=leak, seed=1000 + i, warmup=a, same_words=same)
                    for i in range(SESSIONS)] for a in ARMS}
        rows = {}
        for a in ARMS:
            first = [answers(s, True) for s in runs[a]]
            every = [answers(s, False) for s in runs[a]]
            catch = [caught(s) for s in runs[a]]
            flat = lambda xs: [x for sub in xs for x in sub]  # noqa: E731
            rows[a] = {
                "auc_first_game": pooled_auc(flat(first)),
                "auc_first_game_95ci": boot_ci(first, lambda ss: pooled_auc(flat(ss)), rng),
                "auc_all_games": pooled_auc(flat(every)),
                "auc_all_games_95ci": boot_ci(every, lambda ss: pooled_auc(flat(ss)), rng),
                "mafia_caught_rate": float(np.mean(flat(catch))) if flat(catch) else float("nan"),
                "mafia_games": len(flat(catch)),
                "answers_first_game": len(flat(first)),
            }
        result["by_condition"][f"words={words}, leakiness={leak}"] = rows
        print(f"\nwords {words}, leakiness {leak}")
        print(f"{'warm-up':>8} {'AUC game 1':>24} {'AUC all games':>24} {'mafia caught':>13}")
        for a, r in rows.items():
            ci1, ci2 = r["auc_first_game_95ci"], r["auc_all_games_95ci"]
            print(f"{a:>8} {r['auc_first_game']:.3f} [{ci1[0]:.3f}, {ci1[1]:.3f}]   "
                  f"{r['auc_all_games']:.3f} [{ci2[0]:.3f}, {ci2[1]:.3f}]   {r['mafia_caught_rate']:.0%} of {r['mafia_games']}")
    out = ROOT / "research" / "results" / "warmup_experiment.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
