"""How the AI town thinks: Bayesian beliefs about who the mafia is, and votes chosen to catch them.

Every villager bot holds a probability for each other player being the (single) mafia, starting
uniform. Evidence updates it with Bayes' rule, in log space:

- The lie detector's read on the human's answer, p, from a model whose prior lie rate is `base`.
  If p is calibrated, the likelihood ratio for "the human is the mafia" is
  odds(p) / odds(base), so the update is TRUST * (logit p - logit base). Only the Reader town
  hears this; the Blind town is identical except it never does.
- A weak private gut feeling about each other bot, once a day: a signal s ~ N(HUNCH, 1) if that
  bot is the mafia, N(0, 1) if not, giving log-LR = HUNCH * s - HUNCH^2 / 2. This replaces a hidden
  bonus the bots used to get, with a known accuracy.
- Silence when asked a question: likelihood ratio SILENCE_LR for the human being the mafia.
- Who the human accuses: a villager names the real mafia with probability ACCUSE_HIT, otherwise a
  random other player; the mafia names a random villager.
- Night kills and eliminations: the dead are not the mafia, and who the mafia chose to kill is
  itself evidence (the bot mafia picks uniformly among bots; a human mafia among everyone alive).

Votes are decisions, not reflexes: each villager picks the vote that maximises the probability
that whoever gets eliminated is the mafia, given how everyone else is voting (best responses from
sincere votes, with plurality and random tie-breaks). The mafia bot votes to minimise its own risk,
siding with the town's favourite target. Reader and Blind towns share every random draw.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

TRUST = 1.0         # ceiling on how much a lie-detector read is believed; scaled down by its track record
HUNCH = 0.45        # strength of a bot's daily gut feeling about other bots (signal/noise)
SILENCE_LR = 2.0    # refusing to answer is twice as likely from the mafia
ACCUSE_HIT = 0.4    # chance a villager's accusation names the real mafia
KINDS = ("reader", "blind")
NEG_INF = float("-inf")


def logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


@dataclass
class TownMind:
    human: str
    bots: list[str]
    mafia: str
    seed: int
    trust: float = TRUST
    hunch: float = HUNCH
    alive: dict[str, bool] = field(default_factory=dict)
    # logw[kind][observer][candidate]: unnormalised log posterior that `candidate` is the mafia.
    logw: dict = field(default_factory=dict)
    log: list[dict] = field(default_factory=list)  # every update, for the report and the harness

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        players = [self.human] + self.bots
        self.alive = self.alive or {p: True for p in players}
        self.logw = {k: {o: {c: 0.0 for c in players if c != o} for o in self.bots} for k in KINDS}

    # ---- who thinks what ----------------------------------------------------------------------
    def living(self, exclude: str | None = None) -> list[str]:
        return [p for p, a in self.alive.items() if a and p != exclude]

    def villagers(self) -> list[str]:
        """Bots that are alive and genuinely trying to find the mafia."""
        return [b for b in self.bots if self.alive[b] and b != self.mafia]

    def posterior(self, observer: str, kind: str = "reader") -> dict[str, float]:
        """P(candidate is the mafia) for a villager bot. The mafia bot reports the town's average instead."""
        if observer == self.mafia:
            return self.public(kind, exclude=observer)
        lw = {c: w for c, w in self.logw[kind][observer].items() if self.alive[c]}
        top = max(lw.values())
        if top == NEG_INF:
            return {c: 1 / len(lw) for c in lw}
        ex = {c: math.exp(w - top) for c, w in lw.items()}
        z = sum(ex.values())
        return {c: v / z for c, v in ex.items()}

    def public(self, kind: str = "reader", exclude: str | None = None) -> dict[str, float]:
        """The town's average belief (over living villager bots)."""
        vs = self.villagers()
        cands = self.living(exclude)
        if not vs:
            return {c: 1 / len(cands) for c in cands}
        avg = {c: 0.0 for c in cands}
        for v in vs:
            post = self.posterior(v, kind)
            for c in cands:
                avg[c] += post.get(c, 0.0) / len(vs)
        z = sum(avg.values()) or 1.0
        return {c: p / z for c, p in avg.items()}

    def p_human(self, kind: str = "reader") -> float:
        return self.public(kind).get(self.human, 0.0)

    def top_suspect(self, observer: str, kind: str = "reader") -> str:
        post = self.posterior(observer, kind)
        post.pop(observer, None)
        return max(sorted(post), key=post.get)

    # ---- evidence -----------------------------------------------------------------------------
    def _add(self, kinds, candidate: str, delta: float) -> None:
        for k in kinds:
            for o in self.villagers():
                if candidate in self.logw[k][o]:
                    self.logw[k][o][candidate] += delta

    def observe_read(self, p_lie: float, base: float, reliability: float = 1.0) -> float:
        """A lie-detector read on the human's answer. Reader town only.

        `reliability` in [0, 1] is how much the detector has earned (see detector_reliability)."""
        delta = self.trust * reliability * (logit(p_lie) - logit(base))
        self._add(["reader"], self.human, delta)
        self.log.append({"event": "read", "p_lie": p_lie, "base": base, "delta": delta})
        return delta

    @staticmethod
    def detector_reliability(scored: list[tuple[float, int]], shrink: float = 10.0) -> float:
        """How far the detector has beaten chance on this player, from 0 (coin flip) to 1 (perfect).

        `scored` holds (read made before the label was known, label). The out-of-sample AUC is shrunk
        toward 0.5 by n / (n + shrink), so a few lucky early calls do not buy much trust."""
        pos = [p for p, y in scored if y]
        neg = [p for p, y in scored if not y]
        if not pos or not neg:
            return 0.0
        wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
        auc = wins / (len(pos) * len(neg))
        n = len(scored)
        auc = 0.5 + (auc - 0.5) * n / (n + shrink)
        return max(0.0, min(1.0, 2 * (auc - 0.5)))

    def observe_silence(self) -> None:
        self._add(KINDS, self.human, math.log(SILENCE_LR))
        self.log.append({"event": "silence"})

    def observe_accusation(self, target: str) -> None:
        """The human accused `target`. Likelihood of that under each hypothesis about who the mafia is."""
        n = len(self.living(self.human))           # players the human could name
        if n < 2:
            return
        for k in KINDS:
            for o in self.villagers():
                for c in list(self.logw[k][o]):
                    if not self.alive[c]:
                        continue
                    if c == self.human:
                        like = 1 / n                           # mafia names any of the others, all villagers
                    elif c == target:
                        like = ACCUSE_HIT                      # villager names the real mafia
                    else:
                        like = (1 - ACCUSE_HIT) / max(n - 1, 1)  # villager names someone else
                    self.logw[k][o][c] += math.log(like)
        self.log.append({"event": "accuse", "target": target})

    def observe_hunches(self) -> None:
        """Each villager bot's daily gut feeling about each other living bot. Same draw for both towns."""
        for o in self.villagers():
            for c in self.bots:
                if c == o or not self.alive[c]:
                    continue
                s = self.rng.gauss(self.hunch if c == self.mafia else 0.0, 1.0)
                delta = self.hunch * s - self.hunch ** 2 / 2
                for k in KINDS:
                    self.logw[k][o][c] += delta
        self.log.append({"event": "hunch"})

    def observe_night_kill(self, victim: str) -> None:
        """The mafia killed `victim`. Bot mafia picks among other bots; a human mafia among all bots."""
        before = self.living()
        self.alive[victim] = False
        for k in KINDS:
            for o in self.villagers():
                for c in list(self.logw[k][o]):
                    if c == victim or not (c in before):
                        self.logw[k][o][c] = NEG_INF
                        continue
                    choices = [p for p in before if p != c and (c == self.human or p != self.human)]
                    self.logw[k][o][c] += math.log(1 / len(choices)) if victim in choices else NEG_INF
        self.log.append({"event": "night_kill", "victim": victim})

    def observe_elimination(self, out: str) -> None:
        self.alive[out] = False
        for k in KINDS:
            for o in self.bots:
                if out in self.logw[k][o]:
                    self.logw[k][o][out] = NEG_INF
        self.log.append({"event": "eliminated", "who": out})

    # ---- decisions ----------------------------------------------------------------------------
    @staticmethod
    def outcome(votes: list[str]) -> dict[str, float]:
        """Plurality with uniform tie-breaks: probability each player is eliminated."""
        counts: dict[str, int] = {}
        for v in votes:
            counts[v] = counts.get(v, 0) + 1
        top = max(counts.values())
        tied = [p for p, n in counts.items() if n == top]
        return {p: 1 / len(tied) for p in tied}

    def votes(self, kind: str, human_vote: str | None) -> dict[str, str]:
        voters = [b for b in self.bots if self.alive[b]]
        beliefs = {b: self.posterior(b, kind) for b in voters}

        def sincere(b):
            post = {c: p for c, p in beliefs[b].items() if c != b}
            return max(sorted(post), key=post.get)

        votes = {b: sincere(b) for b in voters}
        if self.mafia in votes:
            votes[self.mafia] = self._mafia_vote(votes, human_vote)
        for _ in range(3):  # best responses until nobody wants to change
            changed = False
            for b in voters:
                if b == self.mafia:
                    best = self._mafia_vote(votes, human_vote)
                else:
                    best = self._best_vote(b, beliefs[b], votes, human_vote)
                if best != votes[b]:
                    votes[b] = best
                    changed = True
            if not changed:
                break
        return votes

    def _ballots(self, votes: dict[str, str], human_vote: str | None, b: str, choice: str) -> list[str]:
        ballots = [v for voter, v in votes.items() if voter != b] + [choice]
        if human_vote is not None and self.alive[self.human]:
            ballots.append(human_vote)
        return ballots

    def _best_vote(self, b: str, post: dict[str, float], votes, human_vote) -> str:
        """Maximise P(the eliminated player is the mafia | my beliefs, everyone else's votes)."""
        def value(choice):
            return sum(prob * post.get(p, 0.0) for p, prob in self.outcome(self._ballots(votes, human_vote, b, choice)).items())
        options = sorted(c for c in self.living(b))
        best = max(value(c) for c in options)
        # Among equally good votes, stay with the sincere favourite (no needless switching).
        good = [c for c in options if value(c) >= best - 1e-9]
        return votes[b] if votes[b] in good else max(good, key=lambda c: post.get(c, 0.0))

    def _mafia_vote(self, votes, human_vote) -> str:
        """Minimise the chance of being eliminated; among safe options, side with the town's favourite."""
        m = self.mafia
        town = self.public("reader", exclude=m)
        def risk(choice):
            return self.outcome(self._ballots(votes, human_vote, m, choice)).get(m, 0.0)
        options = sorted(self.living(m))
        lowest = min(risk(c) for c in options)
        safe = [c for c in options if risk(c) <= lowest + 1e-9]
        return max(safe, key=lambda c: town.get(c, 0.0))

    def mafia_caught_prob(self, votes: dict[str, str], human_vote: str | None) -> float:
        """Probability these votes eliminate the mafia (what the town is trying to maximise)."""
        ballots = list(votes.values()) + ([human_vote] if human_vote and self.alive[self.human] else [])
        return self.outcome(ballots).get(self.mafia, 0.0)
