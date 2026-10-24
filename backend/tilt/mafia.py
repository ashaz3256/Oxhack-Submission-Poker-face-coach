"""Mafia against five AI players, with an Inspector who questions you and may read your vitals.

Each game one player is the mafia; you are the mafia about half the time. Every day the
Inspector asks you a few questions. Because we know your secret role, every answer is
labelled as a lie or the truth the moment you give it.

The town votes on suspicion. Two Inspectors share every rule but one: the Reader adds how
likely your vitals say you just lied; the Blind one only knows how often you lie in general.
The game is played against the one you chose, and the other's votes are computed on the same
day with the same dice, so the extra votes your body drew are measured directly.
"""
from __future__ import annotations

import random
import time
import uuid
from dataclasses import dataclass, field

from .features import TELL_FEATURES, Sample, vector, vitals_features
from .models import OnlineLogit
from .sim import SimPhysiology

YOU = "You"
BOTS = ["Ava", "Ben", "Cleo", "Dev", "Eli"]
QUESTIONS_PER_DAY = 3
READ_WEIGHT = 2.5     # how much one unit of lie probability above base rate moves the town
EVASION_SUSPICION = 0.35
ACCUSED_SUSPICION = 0.2
BOT_MAFIA_HINT = 0.3  # villager bots get a weak sense of a bot mafia, so the town can win


@dataclass
class Option:
    id: str
    text: str
    lie_if_mafia: bool | None    # is this a lie when you are the mafia? None = not a claim
    lie_if_villager: bool | None
    accuses: str | None = None


@dataclass
class Question:
    id: str
    text: str
    options: list[Option]
    asked_at: float


@dataclass
class Answer:
    game: int
    day: int
    question: str
    option: str
    text: str
    t: float
    is_lie: bool | None
    role: str
    features: dict
    p_lie_reader: float | None = None
    p_lie_blind: float | None = None


@dataclass
class DayVote:
    game: int
    day: int
    you_mafia: bool
    votes_on_you_reader: int
    votes_on_you_blind: int
    out_reader: bool
    out_blind: bool


@dataclass
class Game:
    number: int
    mafia: str
    alive: dict[str, bool]
    day: int = 1
    phase: str = "question"   # question | vote | night | over
    question: Question | None = None
    asked_today: int = 0
    asked_ids: list[str] = field(default_factory=list)
    chat: list[dict] = field(default_factory=list)
    winner: str | None = None
    suspicion: dict[str, dict[str, float]] = field(default_factory=dict)
    today_answers: list[Answer] = field(default_factory=list)
    victim: str | None = None

    @property
    def you_mafia(self) -> bool:
        return self.mafia == YOU

    def living(self, exclude: str | None = None) -> list[str]:
        return [p for p, a in self.alive.items() if a and p != exclude]


class MafiaSession:
    def __init__(self, inspector: str = "reader", vitals_source: str = "sim", player: str = "player",
                 leakiness: float = 1.0, mafia_rate: float = 0.5, seed: int | None = None, clock=time.time):
        if inspector not in ("reader", "blind"):
            raise ValueError("inspector must be 'reader' or 'blind'")
        self.id = uuid.uuid4().hex[:12]
        self.inspector = inspector
        self.vitals_source = vitals_source
        self.player = player
        self.mafia_rate = mafia_rate
        self.clock = clock
        self.rng = random.Random(seed)
        self.sim = SimPhysiology(leakiness=leakiness, seed=seed) if vitals_source == "sim" else None
        self.samples: list[Sample] = []
        self.games: list[Game] = []
        self.answers: list[Answer] = []
        self.votes: list[DayVote] = []
        self.lie_model = OnlineLogit(TELL_FEATURES, prior_rate=0.4)
        self.lie_rate = OnlineLogit([], prior_rate=0.4)

    # ---- vitals -------------------------------------------------------------------------------
    def add_samples(self, samples: list[dict]) -> None:
        for s in samples:
            self.samples.append(Sample(t=float(s["t"]), hr=float(s["hr"]), br=s.get("br")))
        self.samples.sort(key=lambda s: s.t)

    def sim_tick(self) -> dict:
        if self.sim is None:
            raise ValueError("session does not use simulated vitals")
        game = self.current
        stress = None
        if game and game.phase == "question":
            # Being questioned is a little stressful for anyone, and much more so with something to hide.
            stress = 1.6 if game.you_mafia else 0.25
        elif game and game.phase == "vote":
            stress = 0.8 if game.you_mafia else 0.2
        sample = self.sim.tick(self.clock(), stress)
        self.add_samples([sample])
        return sample

    # ---- game flow ----------------------------------------------------------------------------
    @property
    def current(self) -> Game | None:
        return self.games[-1] if self.games else None

    def new_game(self) -> Game:
        if self.current and self.current.phase != "over":
            return self.current
        players = [YOU] + BOTS
        mafia = YOU if self.rng.random() < self.mafia_rate else self.rng.choice(BOTS)
        game = Game(number=len(self.games) + 1, mafia=mafia, alive={p: True for p in players})
        game.suspicion = {b: {p: self.rng.gauss(0, 0.3) for p in players if p != b} for b in BOTS}
        role = "the mafia. Don't get caught" if mafia == YOU else "a villager. Find the mafia"
        self._say(game, "Narrator", f"Game {game.number}. You are {role}.")
        self.games.append(game)
        self._ask(game)
        return game

    def answer(self, option_id: str) -> Game:
        game = self._require("question")
        q = game.question
        option = next((o for o in q.options if o.id == option_id), None)
        if option is None:
            raise ValueError(f"unknown option {option_id!r}")
        now = self.clock()
        f = vitals_features(self.samples, now)
        f["decision_s"] = round(now - q.asked_at, 2)
        f = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in f.items()}
        is_lie = option.lie_if_mafia if game.you_mafia else option.lie_if_villager
        a = Answer(game=game.number, day=game.day, question=q.id, option=option.id, text=option.text, t=now,
                   is_lie=is_lie, role="mafia" if game.you_mafia else "villager", features=f)
        x = vector(f, TELL_FEATURES)
        a.p_lie_reader = round(self.lie_model.predict(x), 3)
        a.p_lie_blind = round(self.lie_rate.base_rate(), 3)
        if is_lie is not None:
            # The Inspector learns the truth after the game, like a detective reviewing the tape.
            self.lie_model.add(x, is_lie)
            self.lie_rate.add([], is_lie)
        self.answers.append(a)
        game.today_answers.append(a)
        self._say(game, YOU, option.text)

        for b in game.living(exclude=YOU):
            if option.accuses and option.accuses != b:
                game.suspicion[b][option.accuses] += ACCUSED_SUSPICION
            if is_lie is None:
                game.suspicion[b][YOU] += EVASION_SUSPICION

        game.asked_today += 1
        if game.asked_today < QUESTIONS_PER_DAY:
            self._ask(game)
        else:
            game.question = None
            self._town_talks(game)
            game.phase = "vote"
        return game

    def vote(self, target: str) -> Game:
        game = self._require("vote")
        if target not in game.living(exclude=YOU):
            raise ValueError(f"cannot vote for {target!r}")
        reader_bump, blind_bump = self._read_today(game)
        noise = {b: {p: self.rng.gauss(0, 0.25) for p in game.suspicion[b]} for b in game.living(exclude=YOU)}
        votes_reader = self._bot_votes(game, reader_bump, noise)
        votes_blind = self._bot_votes(game, blind_bump, noise)
        out_reader = self._tally(votes_reader, target)
        out_blind = self._tally(votes_blind, target)
        self.votes.append(DayVote(game=game.number, day=game.day, you_mafia=game.you_mafia,
                                  votes_on_you_reader=sum(v == YOU for v in votes_reader.values()),
                                  votes_on_you_blind=sum(v == YOU for v in votes_blind.values()),
                                  out_reader=out_reader == YOU, out_blind=out_blind == YOU))

        votes = votes_reader if self.inspector == "reader" else votes_blind
        out = out_reader if self.inspector == "reader" else out_blind
        if self.inspector == "reader":
            # Fold today's read into lasting suspicion.
            for b in game.living(exclude=YOU):
                game.suspicion[b][YOU] += reader_bump * 0.5
        for b, v in votes.items():
            self._say(game, b, f"I vote {v}.")
        self._say(game, YOU, f"I vote {target}.")
        game.alive[out] = False
        role = "the mafia" if out == game.mafia else "a villager"
        self._say(game, "Narrator", f"The town eliminates {out}. {out} was {role}.")
        if self._check_over(game):
            return game
        if out == YOU:
            self._play_out(game)
            return game
        game.phase = "night"
        if not game.you_mafia:
            self._night(game, self._bot_kill(game))
        return game

    def kill(self, target: str) -> Game:
        game = self._require("night")
        if not game.you_mafia or target not in game.living(exclude=YOU):
            raise ValueError(f"cannot kill {target!r}")
        self._night(game, target)
        return game

    # ---- internals ----------------------------------------------------------------------------
    def _require(self, phase: str) -> Game:
        game = self.current
        if game is None or game.phase != phase:
            raise ValueError(f"not in the {phase} phase")
        return game

    def _say(self, game: Game, speaker: str, text: str) -> None:
        game.chat.append({"day": game.day, "speaker": speaker, "text": text})

    def _ask(self, game: Game) -> None:
        others = game.living(exclude=YOU)
        pick = self.rng.choice
        bank = []
        bank.append(("are_you", "Are you the mafia?", [
            Option("deny", "No. I'm just a villager.", True, False),
            Option("evade", "Why are you asking me? Ask someone else.", None, None),
        ]))
        if game.victim:
            x = pick(others)
            bank.append(("last_night", f"Where were you last night when {game.victim} died?", [
                Option("asleep", "At home, asleep.", True, False),
                Option(f"saw_{x}", f"Out walking. I saw {x} near {game.victim}'s house.", True, True, accuses=x),
            ]))
        bank.append(("suspect", "Who do you suspect?", [
            Option(f"accuse_{p}", f"I think it's {p}.", True, False, accuses=p) for p in self.rng.sample(others, min(3, len(others)))
        ]))
        y = pick(others)
        bank.append(("trust", f"Do you trust {y}?", [
            Option("trust_yes", f"Yes, {y} is clean.", False, None),
            Option("trust_no", f"No. {y} has been acting strange.", True, None, accuses=y),
        ]))
        if game.asked_today == 0:
            game.asked_ids = []
            qid, text, options = bank[0]
        else:
            fresh = [b for b in bank[1:] if b[0] not in game.asked_ids] or bank[1:]
            qid, text, options = pick(fresh)
        game.asked_ids.append(qid)
        # A villager's opinion of a player is a guess, not a lie; we only know it is a lie when you are the mafia.
        game.question = Question(id=qid, text=text, options=options, asked_at=self.clock())
        game.phase = "question"
        self._say(game, "Inspector", text)

    def _read_today(self, game: Game) -> tuple[float, float]:
        claims = [a for a in game.today_answers if a.p_lie_reader is not None]
        if not claims:
            return 0.0, 0.0
        base = self.lie_rate.base_rate()
        reader = sum(a.p_lie_reader for a in claims) / len(claims) - base
        return READ_WEIGHT * reader, 0.0

    def _bot_votes(self, game: Game, you_bump: float, noise: dict) -> dict[str, str]:
        votes = {}
        for b in game.living(exclude=YOU):
            candidates = game.living(exclude=b)
            if b == game.mafia:
                candidates = [p for p in candidates if p != b]
            def score(p):
                s = game.suspicion[b][p] + noise[b][p]
                if p == YOU:
                    s += you_bump
                if p == game.mafia and b != game.mafia and p != YOU:
                    s += BOT_MAFIA_HINT
                return s
            votes[b] = max(candidates, key=score)
        return votes

    def _tally(self, bot_votes: dict[str, str], your_vote: str) -> str:
        counts: dict[str, int] = {}
        for v in list(bot_votes.values()) + [your_vote]:
            counts[v] = counts.get(v, 0) + 1
        top = max(counts.values())
        tied = sorted(p for p, c in counts.items() if c == top)
        return tied[0] if len(tied) == 1 else random.Random(len(self.votes)).choice(tied)

    def _town_talks(self, game: Game) -> None:
        reader_bump, _ = self._read_today(game)
        for b in game.living(exclude=YOU):
            target = max(game.living(exclude=b), key=lambda p: game.suspicion[b][p] + (reader_bump if p == YOU else 0))
            if target == YOU and self.inspector == "reader" and reader_bump > 0.3:
                line = "You looked nervous answering the Inspector. I'm watching you."
            elif target == YOU:
                line = "Something about your answers doesn't add up."
            else:
                line = self.rng.choice([f"My money's on {target}.", f"{target} has been too quiet.",
                                        f"I don't trust {target}."])
            self._say(game, b, line)
        if self.inspector == "reader" and reader_bump > 0.3:
            self._say(game, "Inspector", "Your pulse jumped on that last answer.")

    def _play_out(self, game: Game) -> None:
        """You were voted out as a villager: the bots finish the game on their own."""
        while game.phase != "over":
            victim = self.rng.choice(game.living(exclude=game.mafia))
            game.alive[victim] = False
            self._say(game, "Narrator", f"Night falls. {victim} is found dead.")
            if self._check_over(game):
                return
            noise = {b: {p: self.rng.gauss(0, 0.25) for p in game.suspicion[b]} for b in game.living()}
            votes = self._bot_votes(game, 0.0, noise)
            counts: dict[str, int] = {}
            for v in votes.values():
                counts[v] = counts.get(v, 0) + 1
            out = max(sorted(counts), key=counts.get)
            game.alive[out] = False
            role = "the mafia" if out == game.mafia else "a villager"
            self._say(game, "Narrator", f"The town eliminates {out}. {out} was {role}.")
            self._check_over(game)

    def _bot_kill(self, game: Game) -> str:
        # The bot mafia spares you so you stay in the game (and keep producing answers).
        return self.rng.choice([p for p in game.living(exclude=game.mafia) if p != YOU])

    def _night(self, game: Game, victim: str) -> None:
        game.alive[victim] = False
        game.victim = victim
        if self._check_over(game):
            return
        game.day += 1
        game.asked_today = 0
        game.today_answers = []
        self._say(game, "Narrator", f"Night falls. In the morning, {victim} is found dead.")
        self._ask(game)

    def _check_over(self, game: Game) -> bool:
        living = game.living()
        if not game.alive[game.mafia]:
            game.winner = "town"
        elif len(living) <= 2:
            game.winner = "mafia"
        else:
            return False
        game.phase = "over"
        game.question = None
        you_won = (game.winner == "mafia") == game.you_mafia
        self._say(game, "Narrator", f"The {game.winner} wins. {'You win!' if you_won else 'You lose.'}")
        return True

    # ---- views --------------------------------------------------------------------------------
    def view(self) -> dict:
        game = self.current
        if game is None:
            return {"game": None}
        over = game.phase == "over"
        players = [{"name": p, "alive": a,
                    "role": ("mafia" if p == game.mafia else "villager") if (over or not a or p == YOU) else None}
                   for p, a in game.alive.items()]
        q = game.question
        finished = [g for g in self.games if g.phase == "over"]
        wins = sum((g.winner == "mafia") == g.you_mafia for g in finished)
        return {
            "game": {
                "number": game.number,
                "day": game.day,
                "phase": game.phase,
                "you_mafia": game.you_mafia,
                "players": players,
                "question": None if q is None else {"id": q.id, "text": q.text,
                                                    "options": [{"id": o.id, "text": o.text} for o in q.options]},
                "vote_options": game.living(exclude=YOU) if game.phase in ("vote", "night") else [],
                "chat": game.chat,
                "winner": game.winner,
            },
            "record": {"games": len(finished), "wins": wins},
        }
