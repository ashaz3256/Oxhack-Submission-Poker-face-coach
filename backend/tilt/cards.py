"""Card helpers and Monte Carlo equity, built on treys."""
from __future__ import annotations

import random
from functools import lru_cache

from treys import Card, Evaluator

EVALUATOR = Evaluator()
RANKS = "23456789TJQKA"
SUITS = "shdc"
FULL_DECK: tuple[int, ...] = tuple(Card.new(r + s) for r in RANKS for s in SUITS)


def to_str(cards: list[int]) -> list[str]:
    return [Card.int_to_str(c) for c in cards]


def from_str(cards: list[str]) -> list[int]:
    return [Card.new(c) for c in cards]


def _remaining(dead: list[int]) -> list[int]:
    dead_set = set(dead)
    return [c for c in FULL_DECK if c not in dead_set]


def showdown(board: list[int], a: list[int], b: list[int]) -> int:
    """1 if a wins, -1 if b wins, 0 on a chop. Board must have 5 cards."""
    ra = EVALUATOR.evaluate(board, a)
    rb = EVALUATOR.evaluate(board, b)
    return (ra < rb) - (ra > rb)


def equity_vs_random(hole: list[int], board: list[int], samples: int = 300, rng: random.Random | None = None) -> float:
    """Probability that `hole` beats a uniformly random opponent hand by the river."""
    return _equity_cached(tuple(sorted(hole)), tuple(sorted(board)), samples) if rng is None else _equity(hole, board, samples, rng)


@lru_cache(maxsize=50_000)
def _equity_cached(hole: tuple[int, ...], board: tuple[int, ...], samples: int) -> float:
    seed = hash((hole, board)) & 0xFFFFFFFF
    return _equity(list(hole), list(board), samples, random.Random(seed))


def _equity(hole: list[int], board: list[int], samples: int, rng: random.Random) -> float:
    deck = _remaining(hole + board)
    need = 5 - len(board)
    score = 0.0
    for _ in range(samples):
        draw = rng.sample(deck, 2 + need)
        opp, runout = draw[:2], board + draw[2:]
        score += (showdown(runout, hole, opp) + 1) / 2
    return score / samples


def split_equity(hero_range_hole: list[int], board: list[int], dead: list[int], strong_cut: float,
                 samples: int = 120, inner: int = 80, rng: random.Random | None = None) -> tuple[float, float]:
    """Equity of `hero_range_hole` (the bot's cards) against villain hands split by strength.

    Villain hands are sampled at random, labelled weak (equity vs random < strong_cut) or
    strong, and the bot's showdown equity is averaged within each group. Returns
    (equity vs weak hands, equity vs strong hands).
    """
    rng = rng or random.Random()
    deck = _remaining(hero_range_hole + board + dead)
    need = 5 - len(board)
    totals = {True: [0.0, 0], False: [0.0, 0]}
    for _ in range(samples):
        villain = rng.sample(deck, 2)
        strong = equity_vs_random(villain, board, samples=inner) >= strong_cut
        rest = [c for c in deck if c not in villain]
        runout = board + rng.sample(rest, need)
        totals[strong][0] += (showdown(runout, hero_range_hole, villain) + 1) / 2
        totals[strong][1] += 1
    weak = totals[False][0] / totals[False][1] if totals[False][1] else 0.5
    strong = totals[True][0] / totals[True][1] if totals[True][1] else 0.5
    return weak, strong
