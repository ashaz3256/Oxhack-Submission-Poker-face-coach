"""Heads-up hold'em against a bot, simplified so every hand produces several labelled decisions.

Each hand both players ante 1bb. On every street the hero acts first: check, bet half pot or
bet pot. A hero bet is met with call or fold (no raises). If the hero checks, the bot may bet
half pot and the hero calls or folds. Hands that survive the river go to showdown.

Two opponents share the same betting logic and differ in one thing only: when the hero bets,
the Honest bot assumes the hero bluffs at their historical rate, while the Shark asks its tell
model how likely this particular bet is a bluff, given the hero's vitals. Both responses are
computed on every hero bet, so the chips your tells cost you are measured on the same hands.
"""
from __future__ import annotations

import random
import time
import uuid
from dataclasses import dataclass, field

from . import cards
from .features import TELL_FEATURES, TILT_FEATURES, Sample, vector, vitals_features
from .models import OnlineLogit
from .sim import SimPhysiology

STREETS = ["preflop", "flop", "turn", "river"]
BOARD_SIZE = [0, 3, 4, 5]
ANTE = 1.0
BLUFF_CUT = 0.5          # a hero bet with equity vs a random hand below this is a bluff
BAD_CALL_MARGIN = 0.05   # calling with equity this far below the pot odds is a bad decision
BAD_FOLD_MARGIN = 0.12   # folding with equity this far above the pot odds is a bad decision
BOT_VALUE_BET = 0.62
BOT_BLUFF_FREQ = 0.12


@dataclass
class Hand:
    id: str
    number: int
    hero: list[int]
    bot: list[int]
    board_full: list[int]
    started_at: float
    street: int = 0
    pot: float = 2 * ANTE
    hero_in: float = ANTE
    bot_in: float = ANTE
    phase: str = "hero_act"   # hero_act | hero_facing_bet | done
    to_call: float = 0.0
    street_started_at: float = 0.0
    log: list[str] = field(default_factory=list)
    result_bb: float | None = None
    showdown: bool = False

    @property
    def board(self) -> list[int]:
        return self.board_full[: BOARD_SIZE[self.street]]


@dataclass
class Decision:
    hand_id: str
    hand_number: int
    street: str
    t: float
    action: str
    amount: float
    pot_before: float
    to_call: float
    strength: float
    pot_odds: float | None
    is_bluff: bool | None
    is_bad: bool | None
    features: dict
    p_bluff_shark: float | None = None
    p_bluff_honest: float | None = None
    response_shark: str | None = None
    response_honest: str | None = None
    ev_vs_shark: float | None = None
    ev_vs_honest: float | None = None
    p_bad: float | None = None


def bot_call_decision(eq_weak: float, eq_strong: float, pot_before: float, bet: float, p_bluff: float) -> str:
    """Call when the bot's equity, weighted by how likely the bet is a bluff, beats the pot odds."""
    equity = p_bluff * eq_weak + (1 - p_bluff) * eq_strong
    return "call" if equity > bet / (pot_before + 2 * bet) else "fold"


def hero_ev_after_bet(hand: Hand, pot_before: float, bet: float, response: str) -> float:
    """Hero's expected chips back from this point, assuming a check-down if called."""
    if response == "fold":
        return pot_before
    equity = _true_equity(hand.hero, hand.bot, hand.board)
    return equity * (pot_before + 2 * bet) - bet


def _true_equity(hero: list[int], bot: list[int], board: list[int], samples: int = 400) -> float:
    if len(board) == 5:
        return (cards.showdown(board, hero, bot) + 1) / 2
    rng = random.Random(hash((tuple(hero), tuple(bot), tuple(board))))
    deck = [c for c in cards.FULL_DECK if c not in hero + bot + board]
    need = 5 - len(board)
    total = 0.0
    for _ in range(samples):
        runout = board + rng.sample(deck, need)
        total += (cards.showdown(runout, hero, bot) + 1) / 2
    return total / samples


class Session:
    def __init__(self, opponent: str = "shark", vitals_source: str = "sim", player: str = "player",
                 leakiness: float = 1.0, seed: int | None = None, clock=time.time):
        if opponent not in ("shark", "honest"):
            raise ValueError("opponent must be 'shark' or 'honest'")
        self.id = uuid.uuid4().hex[:12]
        self.opponent = opponent
        self.vitals_source = vitals_source
        self.player = player
        self.clock = clock
        self.rng = random.Random(seed)
        self.sim = SimPhysiology(leakiness=leakiness, seed=seed) if vitals_source == "sim" else None
        self.samples: list[Sample] = []
        self.hands: list[Hand] = []
        self.decisions: list[Decision] = []
        self.tell_model = OnlineLogit(TELL_FEATURES, prior_rate=0.3)
        self.bluff_rate = OnlineLogit([], prior_rate=0.3)  # features unused: base rate only
        self.tilt_model = OnlineLogit(TILT_FEATURES, prior_rate=0.2)
        self.net_bb = 0.0
        self.peak_bb = 0.0

    # ---- vitals -------------------------------------------------------------------------------
    def add_samples(self, samples: list[dict]) -> None:
        for s in samples:
            self.samples.append(Sample(t=float(s["t"]), hr=float(s["hr"]), br=s.get("br")))
        self.samples.sort(key=lambda s: s.t)

    def sim_tick(self) -> dict:
        if self.sim is None:
            raise ValueError("session does not use simulated vitals")
        hand = self.current_hand
        live = hand is not None and hand.phase != "done"
        strength = cards.equity_vs_random(hand.hero, hand.board) if live else None
        sample = self.sim.tick(self.clock(), strength, hand.pot if live else 0.0)
        self.add_samples([sample])
        return sample

    # ---- game flow ----------------------------------------------------------------------------
    @property
    def current_hand(self) -> Hand | None:
        return self.hands[-1] if self.hands else None

    def new_hand(self) -> Hand:
        if self.current_hand and self.current_hand.phase != "done":
            return self.current_hand
        deck = list(cards.FULL_DECK)
        self.rng.shuffle(deck)
        now = self.clock()
        hand = Hand(id=uuid.uuid4().hex[:12], number=len(self.hands) + 1, hero=deck[0:2], bot=deck[2:4],
                    board_full=deck[4:9], started_at=now, street_started_at=now)
        self.hands.append(hand)
        return hand

    def act(self, action: str) -> Hand:
        hand = self.current_hand
        if hand is None or hand.phase == "done":
            raise ValueError("no hand in progress")
        legal = self.legal_actions(hand)
        if action not in legal:
            raise ValueError(f"illegal action {action!r}; legal: {legal}")
        now = self.clock()
        strength = cards.equity_vs_random(hand.hero, hand.board)
        features = self._features(hand, now)
        street = STREETS[hand.street]

        if hand.phase == "hero_act":
            if action == "check":
                self._record(hand, now, action, 0.0, strength, None, None, None, features)
                hand.log.append(f"{street}: you check")
                self._bot_turn(hand)
            else:
                pot_before = hand.pot
                bet = round(pot_before * (0.5 if action == "bet_half" else 1.0), 2)
                decision = self._record(hand, now, action, bet, strength, None, strength < BLUFF_CUT, None, features)
                self._resolve_hero_bet(hand, decision, pot_before, bet)
        else:  # hero_facing_bet
            bet = hand.to_call
            pot_odds = bet / (hand.pot + bet)
            if action == "call":
                bad = strength < pot_odds - BAD_CALL_MARGIN
            else:
                bad = strength > pot_odds + BAD_FOLD_MARGIN
            decision = self._record(hand, now, action, bet if action == "call" else 0.0, strength, pot_odds, None, bad, features)
            decision.p_bad = self.tilt_model.predict(vector(features, TILT_FEATURES))
            self.tilt_model.add(vector(features, TILT_FEATURES), bad)
            if action == "call":
                hand.hero_in += bet
                hand.pot += bet
                hand.to_call = 0.0
                hand.log.append(f"{street}: you call {bet:g}")
                self._next_street(hand)
            else:
                hand.log.append(f"{street}: you fold")
                self._finish(hand, -hand.hero_in)
        return hand

    def legal_actions(self, hand: Hand | None = None) -> list[str]:
        hand = hand or self.current_hand
        if hand is None or hand.phase == "done":
            return []
        return ["check", "bet_half", "bet_pot"] if hand.phase == "hero_act" else ["call", "fold"]

    # ---- internals ----------------------------------------------------------------------------
    def _features(self, hand: Hand, now: float) -> dict:
        f = vitals_features(self.samples, now)
        f["decision_s"] = round(now - hand.street_started_at, 2)
        finished = [h for h in self.hands if h.result_bb is not None]
        f["last_result_bb"] = finished[-1].result_bb if finished else 0.0
        f["drawdown_bb"] = self.peak_bb - self.net_bb
        return f

    def _record(self, hand, now, action, amount, strength, pot_odds, is_bluff, is_bad, features) -> Decision:
        d = Decision(hand_id=hand.id, hand_number=hand.number, street=STREETS[hand.street], t=now, action=action,
                     amount=amount, pot_before=hand.pot, to_call=hand.to_call, strength=round(strength, 3),
                     pot_odds=None if pot_odds is None else round(pot_odds, 3), is_bluff=is_bluff, is_bad=is_bad,
                     features={k: (round(v, 3) if isinstance(v, float) else v) for k, v in features.items()})
        self.decisions.append(d)
        return d

    def _resolve_hero_bet(self, hand: Hand, d: Decision, pot_before: float, bet: float) -> None:
        x = vector(d.features, TELL_FEATURES)
        d.p_bluff_shark = round(self.tell_model.predict(x), 3)
        d.p_bluff_honest = round(self.bluff_rate.base_rate(), 3)
        # One equity estimate for both bots, so the only difference between them is the bluff read.
        eq_weak, eq_strong = cards.split_equity(hand.bot, hand.board, [], BLUFF_CUT, rng=random.Random(self.rng.random()))
        d.response_shark = bot_call_decision(eq_weak, eq_strong, pot_before, bet, d.p_bluff_shark)
        d.response_honest = bot_call_decision(eq_weak, eq_strong, pot_before, bet, d.p_bluff_honest)
        d.ev_vs_shark = round(hero_ev_after_bet(hand, pot_before, bet, d.response_shark), 3)
        d.ev_vs_honest = round(hero_ev_after_bet(hand, pot_before, bet, d.response_honest), 3)
        # Learn after acting: the Shark sees the hero's cards once the hand is logged.
        self.tell_model.add(x, bool(d.is_bluff))
        self.bluff_rate.add([], bool(d.is_bluff))

        street = STREETS[hand.street]
        hand.hero_in += bet
        hand.pot += bet
        hand.log.append(f"{street}: you bet {bet:g}")
        response = d.response_shark if self.opponent == "shark" else d.response_honest
        if response == "fold":
            hand.log.append(f"{street}: bot folds")
            self._finish(hand, hand.bot_in)
        else:
            hand.bot_in += bet
            hand.pot += bet
            hand.log.append(f"{street}: bot calls")
            self._next_street(hand)

    def _bot_turn(self, hand: Hand) -> None:
        equity = cards.equity_vs_random(hand.bot, hand.board)
        if equity > BOT_VALUE_BET or self.rng.random() < BOT_BLUFF_FREQ:
            bet = round(hand.pot * 0.5, 2)
            hand.bot_in += bet
            hand.pot += bet
            hand.to_call = bet
            hand.phase = "hero_facing_bet"
            hand.log.append(f"{STREETS[hand.street]}: bot bets {bet:g}")
        else:
            hand.log.append(f"{STREETS[hand.street]}: bot checks")
            self._next_street(hand)

    def _next_street(self, hand: Hand) -> None:
        if hand.street == 3:
            outcome = cards.showdown(hand.board_full, hand.hero, hand.bot)
            hand.showdown = True
            result = {1: hand.bot_in, -1: -hand.hero_in, 0: (hand.bot_in - hand.hero_in) / 2}[outcome]
            hand.log.append({1: "showdown: you win", -1: "showdown: bot wins", 0: "showdown: split pot"}[outcome])
            self._finish(hand, result)
            return
        hand.street += 1
        hand.phase = "hero_act"
        hand.street_started_at = self.clock()

    def _finish(self, hand: Hand, result_bb: float) -> None:
        hand.phase = "done"
        hand.result_bb = round(result_bb, 2)
        self.net_bb += hand.result_bb
        self.peak_bb = max(self.peak_bb, self.net_bb)
        if self.sim:
            self.sim.on_hand_result(hand.result_bb)

    # ---- views --------------------------------------------------------------------------------
    def hand_view(self, hand: Hand | None = None) -> dict:
        hand = hand or self.current_hand
        if hand is None:
            return {"hand": None, "net_bb": self.net_bb}
        done = hand.phase == "done"
        return {
            "hand": {
                "id": hand.id,
                "number": hand.number,
                "street": STREETS[hand.street],
                "hero": cards.to_str(hand.hero),
                "bot": cards.to_str(hand.bot) if done else None,
                "board": cards.to_str(hand.board_full if done and hand.showdown else hand.board),
                "pot": round(hand.pot, 2),
                "to_call": hand.to_call,
                "phase": hand.phase,
                "legal": self.legal_actions(hand),
                "log": hand.log,
                "result_bb": hand.result_bb,
            },
            "net_bb": round(self.net_bb, 2),
            "hands_played": sum(1 for h in self.hands if h.result_bb is not None),
        }
