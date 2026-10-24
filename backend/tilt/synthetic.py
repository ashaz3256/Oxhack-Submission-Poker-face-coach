"""A scripted hero for tests, demos and pipeline checks before real hands exist."""
from __future__ import annotations

import random

from . import cards
from .game import Session


class FakeClock:
    def __init__(self, start: float = 1_700_000_000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def scripted_action(session: Session, rng: random.Random) -> str:
    hand = session.current_hand
    strength = cards.equity_vs_random(hand.hero, hand.board)
    if hand.phase == "hero_facing_bet":
        pot_odds = hand.to_call / (hand.pot + hand.to_call)
        return "call" if strength + rng.gauss(0, 0.08) > pot_odds else "fold"
    if strength > 0.62:
        return "bet_pot" if rng.random() < 0.5 else "bet_half"
    if strength < 0.35 and rng.random() < 0.45:
        return "bet_pot" if rng.random() < 0.5 else "bet_half"
    return "check"


def play(n_hands: int = 60, leakiness: float = 1.0, opponent: str = "shark", seed: int = 7) -> Session:
    """Play `n_hands` with simulated vitals ticking once a second while the hero thinks."""
    clock = FakeClock()
    rng = random.Random(seed)
    session = Session(opponent=opponent, vitals_source="sim", leakiness=leakiness, seed=seed, clock=clock)
    for _ in range(45):  # resting baseline
        clock.advance(1)
        session.sim_tick()
    for _ in range(n_hands):
        session.new_hand()
        while session.current_hand.phase != "done":
            for _ in range(rng.randint(4, 9)):
                clock.advance(1)
                session.sim_tick()
            session.act(scripted_action(session, rng))
        for _ in range(3):
            clock.advance(1)
            session.sim_tick()
    return session


def play_mafia(n_games: int = 12, leakiness: float = 1.0, inspector: str = "reader", seed: int = 7):
    """Play `n_games` of Mafia with a scripted player and simulated vitals."""
    from .mafia import YOU, MafiaSession

    clock = FakeClock()
    rng = random.Random(seed)
    session = MafiaSession(inspector=inspector, vitals_source="sim", leakiness=leakiness, seed=seed, clock=clock)

    def wait(lo: int, hi: int) -> None:
        for _ in range(rng.randint(lo, hi)):
            clock.advance(1)
            session.sim_tick()

    wait(45, 45)  # resting baseline
    for _ in range(n_games):
        game = session.new_game()
        while game.phase != "over":
            if game.phase == "question":
                wait(3, 8)
                options = game.question.options
                pick = options[0] if game.question.id == "are_you" and rng.random() < 0.85 else rng.choice(options)
                session.answer(pick.id)
            elif game.phase == "vote":
                wait(2, 4)
                session.vote(rng.choice(game.living(exclude=YOU)))
            elif game.phase == "night":
                wait(1, 2)
                session.kill(rng.choice(game.living(exclude=YOU)))
        wait(3, 3)
    return session
