"""Spoken Mafia: you and five AI players on a call. Everything you say is analysed for lies.

Flow of a game:
  pick   - everyone draws a card; you say a number from one to six
  talk   - AI players discuss and question you twice a day; you answer out loud
  vote   - you say a name; the AIs vote too
  night  - if you are the mafia you say who to eliminate; otherwise the AI mafia acts
  over   - say "play again"

As the mafia, everything you say in the discussion serves a lie, so each answer of three words
or more is labelled a lie when you are the mafia and the truth when you are not. Those labels are
only released to the lie model when the game ends: within a game every answer has the same label,
so learning them as they come would teach the detector your role rather than your tells (a player
with no tells at all was read perfectly by the third answer). For the same reason the town only
trusts the detector on what it got right in earlier games and the warm-up.

The AI town thinks in probabilities (beliefs.py): each villager bot holds P(mafia) for every
player and updates it with Bayes' rule from the lie detector's reads, silences, accusations, kills
and a weak daily gut feeling about the other bots, then votes to maximise the chance the mafia goes
out. Every vote is also replayed by a "blind" town that never hears the lie detector (same dice),
so the report can say how often your body got you caught.
"""
from __future__ import annotations

import random
import time
import uuid
from dataclasses import dataclass, field

from . import dialogue
from .cues import FACE_SCORE, describe
from .beliefs import TownMind
from .face_model import FaceScorer
from .language import find_names, last_name, parse_number, wants_again
from .liemodel import LieModel
from .presage import PresageVitals

YOU = "You"
BOTS = ["Ava", "Ben", "Cleo", "Dev", "Eli"]
PLAYERS = [YOU] + BOTS
NARRATOR = "Narrator"
MIN_CLAIM_WORDS = 3
QUESTIONS_PER_DAY = 2


@dataclass
class Utterance:
    game: int
    day: int
    t: float
    kind: str                 # calibration | answer
    question: str
    transcript: str
    role: str
    raw: dict
    features: dict
    is_lie: bool | None
    p_lie: float | None = None
    p_blind: float | None = None
    cues: list = field(default_factory=list)
    llm: float | None = None   # the language judge's lie score, 0..1


@dataclass
class DayVote:
    game: int
    day: int
    you_mafia: bool
    votes_on_you_reader: int
    votes_on_you_blind: int
    out_reader: bool
    out_blind: bool
    p_you_reader: float | None = None     # town's average P(you are the mafia) when it voted
    p_you_blind: float | None = None
    p_catch_reader: float | None = None   # chance the votes eliminate the real mafia
    p_catch_blind: float | None = None


@dataclass
class Game:
    number: int
    alive: dict[str, bool]
    mafia: str | None = None
    cards: list[str] = field(default_factory=list)       # card i -> player who drew it
    phase: str = "pick"
    day: int = 1
    asked_today: int = 0
    awaiting: dict | None = None
    retries: int = 0
    winner: str | None = None
    victim: str | None = None
    mind: TownMind | None = None   # the AI town's beliefs (set once roles are dealt)
    log: list[dict] = field(default_factory=list)   # answers, votes and kills, for the breakdown

    @property
    def you_mafia(self) -> bool:
        return self.mafia == YOU

    def living(self, exclude: str | None = None) -> list[str]:
        return [p for p, a in self.alive.items() if a and p != exclude]


class MafiaSession:
    def __init__(self, player: str = "player", mafia_rate: float = 0.5, seed: int | None = None, clock=time.time):
        self.id = uuid.uuid4().hex[:12]
        self.player = player
        self.mafia_rate = mafia_rate
        self.clock = clock
        self.rng = random.Random(seed)
        self.model = LieModel()
        self.face = FaceScorer()
        self.face_frames: list[list[float]] | None = None  # this turn's per-frame face primitives, set by the API
        self.utterances: list[Utterance] = []
        self.pending_labels: list[tuple] = []  # this game's labelled answers, learned when it ends
        self.votes: list[DayVote] = []
        self.games: list[Game] = []
        self.chat: list[dict] = []
        self.outbox: list[dict] = []
        self.last_read: dict | None = None
        self.presage = PresageVitals()
        self.belief_log: list[dict] = []   # every villager bot's beliefs at each vote, for the report

    # ---- talking ------------------------------------------------------------------------------
    def _say(self, speaker: str, text: str, event: dict | None = None) -> None:
        game = self.current
        line = {"speaker": speaker, "text": text, "game": game.number if game else 0, "day": game.day if game else 0}
        if event:
            line["event"] = event   # tells the call screen to play an animation with this line
        self.chat.append(line)
        self.outbox.append(line)

    def drain(self) -> list[dict]:
        out, self.outbox = self.outbox, []
        return out

    def _await(self, kind: str, prompt: str, asker: str = NARRATOR) -> None:
        self.current.awaiting = {"kind": kind, "prompt": prompt, "asker": asker, "since": self.clock()}

    def _context(self, game: Game, extra: dict | None = None) -> dict:
        recent = [f"{l['speaker']}: {l['text']}" for l in self.chat[-12:]]
        ctx = {"day": game.day, "alive": game.living(), "dead": [p for p in PLAYERS if not game.alive[p]],
               "last_victim": game.victim, "recent_conversation": recent,
               "human_is_called": "You (address them as 'you')"}
        if game.mafia and game.mafia != YOU:
            ctx["mafia_secret"] = f"{game.mafia} is the mafia and should deflect suspicion without revealing it"
        if game.mind:
            ctx["each_players_private_odds_that_someone_is_mafia_percent"] = {
                b: {p: round(100 * v) for p, v in game.mind.posterior(b).items() if p != b}
                for b in game.living(exclude=YOU)}
            ctx["how_to_use_odds"] = ("Let each speaker's suspicions follow their own odds; the analyst may quote "
                                      "a rough percentage. Never reveal the odds table itself.")
        return {**ctx, **(extra or {})}

    # ---- calibration --------------------------------------------------------------------------
    def calibrate(self, transcript: str, raw: dict, is_lie: bool = False, question: str = "") -> dict:
        """Practice round before the game: honest answers set your baseline, instructed lies teach the
        detector what *your* lying looks like."""
        raw = self._with_face(self._with_vitals(raw))
        llm = dialogue.judge_lie(question, transcript, []) if transcript else None
        p, contrib, f = self.model.predict(raw, transcript, llm)
        u = Utterance(game=0, day=0, t=self.clock(), kind="calibration", question=question, transcript=transcript,
                      role="practice", raw=raw, features={k: round(v, 3) for k, v in f.items()}, is_lie=is_lie,
                      p_lie=round(p, 3), p_blind=round(self.model.base_rate, 3), cues=self._cues(contrib, f), llm=llm)
        self.utterances.append(u)
        self.model.add(raw, transcript, llm, is_lie, honest_baseline=not is_lie)
        self.last_read = {"p_lie": round(p, 3), "cues": u.cues, "top": None, "features": u.features, "labelled": is_lie}
        return {"p_lie": round(p, 3), "was_lie": is_lie, "cues": u.cues,
                "accuracy": self.model.cross_validated_accuracy(), "samples": len(self.model.samples)}

    def _with_vitals(self, raw: dict) -> dict:
        """Presage's heart and breathing rate, when fresh, replace our own webcam estimate."""
        return {**raw, **self.presage.fresh()}

    @staticmethod
    def _cues(contrib: dict, f: dict) -> list[str]:
        ranked = sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)
        return [describe(k, f[k]) for k, v in ranked if v > 0.05][:3]

    # ---- game flow ----------------------------------------------------------------------------
    @property
    def current(self) -> Game | None:
        return self.games[-1] if self.games else None

    def new_game(self) -> Game:
        if self.current and self.current.phase != "over":
            return self.current
        game = Game(number=len(self.games) + 1, alive={p: True for p in PLAYERS})
        self.games.append(game)
        self._say(NARRATOR, "Right. Six cards, face down. One of them's the mafia. Pick one. Say a number, or tap a card.",
                  event={"type": "intro", "game": game.number})
        self._await("pick", "Say a number from one to six")
        return game

    def speak(self, transcript: str, raw: dict | None = None) -> None:
        """Whatever you just said, routed by what the game was waiting for."""
        game = self.current
        if game is None or game.awaiting is None:
            raise ValueError("the game is not waiting for you")
        transcript = (transcript or "").strip()
        raw = raw or {}
        kind = game.awaiting["kind"]
        {"pick": self._on_pick, "answer": self._on_answer, "vote": self._on_vote, "kill": self._on_kill,
         "again": self._on_again}[kind](game, transcript, raw)

    def _on_pick(self, game: Game, transcript: str, raw: dict) -> None:
        n = parse_number(transcript)
        if n is None and game.retries < 1 and transcript:
            game.retries += 1
            self._say(NARRATOR, "Sorry, which card? One to six.")
            self._await("pick", "Say a number from one to six")
            return
        game.retries = 0
        n = n or self.rng.randint(1, 6)
        if transcript:
            self._say(YOU, transcript)
        you_mafia = self.rng.random() < self.mafia_rate
        game.mafia = YOU if you_mafia else self.rng.choice(BOTS)
        order = BOTS[:]
        self.rng.shuffle(order)
        game.cards = order[: n - 1] + [YOU] + order[n - 1:]
        game.mind = TownMind(human=YOU, bots=list(BOTS), mafia=game.mafia, seed=self.rng.randrange(2**31))
        role = "the mafia. Lie your way to the end" if you_mafia else "a villager. Find the mafia before it finds you"
        self._say(NARRATOR, f"You drew card {n}. You are {role}.")
        self._start_day(game)

    def _start_day(self, game: Game) -> None:
        game.phase = "talk"
        game.asked_today = 0
        game.mind.observe_hunches()
        if game.day == 1:
            self._say(NARRATOR, "Day one. Everyone's got their card, and somebody here is lying. Go on then. Talk.")
        else:
            self._say(NARRATOR, f"Morning. {game.victim} didn't make it through the night. So. Who did it?",
                      event={"type": "death", "who": game.victim, "how": "night"})
        speakers = self.rng.sample(game.living(exclude=YOU), min(3, len(game.living(exclude=YOU))))
        self._round(game, speakers, opening=True)

    def _round(self, game: Game, speakers: list[str], opening: bool, read: dict | None = None) -> None:
        """Some AI players talk; the last one puts a question to you."""
        others = game.living(exclude=YOU)

        def fallback():
            lines = []
            for i, s in enumerate(speakers):
                if i == len(speakers) - 1:
                    asked = [u.question for u in self.utterances if u.game == game.number]
                    q = dialogue.template_question(self.rng, [p for p in others if p != s] or others, game.day, asked)
                    lines.append({"speaker": s, "text": q})
                elif read is not None and i == 0:
                    lines.append({"speaker": s, "text": dialogue.template_reaction(self.rng, read["p_lie"], read["top"])})
                elif opening and i == 0 and game.day == 1:
                    lines.append({"speaker": s, "text": self.rng.choice(dialogue.OPENERS)})
                else:
                    target = game.mind.top_suspect(s)
                    target = "you" if target == YOU else target
                    lines.append({"speaker": s, "text": self.rng.choice(dialogue.SUSPECT).format(x=target)})
            return lines

        extra = {}
        if read is not None:
            extra["lie_detector_on_human_last_answer"] = {
                "probability_lying": read["p_lie"], "strongest_cues": read["cues"][:2]}
        task = ("Open the day's discussion: react to the situation, voice a suspicion, "
                if opening else "React to what the human just said (first speaker reacts directly, using the lie "
                "detector reading if it is high, e.g. mention their voice, eyes or pauses), ")
        task += f"and the last speaker ({speakers[-1]}) asks the human one direct, pointed question."
        lines = dialogue.ai_lines(speakers=speakers, task=task, game=self._context(game, extra), rng=self.rng,
                                  fallback=fallback)
        for line in lines:
            self._say(line["speaker"], line["text"])
        self._await("answer", lines[-1]["text"], asker=speakers[-1])

    def _on_answer(self, game: Game, transcript: str, raw: dict) -> None:
        question = game.awaiting["prompt"]
        asker = game.awaiting["asker"]
        if not transcript:
            game.mind.observe_silence()
            self._say(asker, "Nothing to say? That's telling.")
            game.log.append({"type": "silent", "day": game.day, "question": question})
            read = None
        else:
            self._say(YOU, transcript)
            read = self._analyse(game, question, transcript, raw)
            for name in find_names(transcript, game.living(exclude=YOU)):
                game.mind.observe_accusation(name)
        game.asked_today += 1
        if game.asked_today < QUESTIONS_PER_DAY:
            others = [p for p in game.living(exclude=YOU) if p != asker] or game.living(exclude=YOU)
            speakers = [asker] + self.rng.sample(others, min(1, len(others)))
            self._round(game, speakers, opening=False, read=read)
        else:
            if read is not None:
                self._say(asker, dialogue.template_reaction(self.rng, read["p_lie"], read["top"]))
            game.phase = "vote"
            self._say(NARRATOR, "Okay, time to vote. Say the name of who you want out.")
            self._await("vote", "Say a name: " + ", ".join(game.living(exclude=YOU)))

    def _with_face(self, raw: dict) -> dict:
        """Attach this turn's dataset-trained face score to the measurements the lie model stores."""
        score, _ = self.face.score(self._take_face_frames())
        return {**raw, FACE_SCORE: score}

    def _take_face_frames(self) -> list[list[float]] | None:
        frames, self.face_frames = self.face_frames, None
        return frames

    def _analyse(self, game: Game, question: str, transcript: str, raw: dict) -> dict:
        raw = self._with_face(self._with_vitals(raw))
        recent = [f"{l['speaker']}: {l['text']}" for l in self.chat[-10:]]
        llm = dialogue.judge_lie(question, transcript, recent)
        p, contrib, f = self.model.predict(raw, transcript, llm)
        base = self.model.base_rate
        ranked = sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)
        top = [k for k, v in ranked if v > 0.05][:3]
        n_words = len(transcript.split())
        is_lie = game.you_mafia if n_words >= MIN_CLAIM_WORDS else None
        u = Utterance(game=game.number, day=game.day, t=self.clock(), kind="answer", question=question,
                      transcript=transcript, role="mafia" if game.you_mafia else "villager", raw=raw,
                      features={k: round(v, 3) for k, v in f.items()}, is_lie=is_lie, p_lie=round(p, 3),
                      p_blind=round(base, 3), cues=[describe(k, f[k]) for k in top], llm=llm)
        self.utterances.append(u)
        if is_lie is not None:
            self.pending_labels.append((raw, transcript, llm, is_lie))
        scored = [(x.p_lie, int(x.is_lie)) for x in self.utterances
                  if x.game != game.number and x.is_lie is not None and x.p_lie is not None]
        reliability = TownMind.detector_reliability(scored)
        delta = game.mind.observe_read(p, base, reliability)
        self.last_read_reliability = reliability
        game.log.append({"type": "answer", "day": game.day, "utterance": len(self.utterances) - 1,
                         "bump": round(delta, 3), "rank": self._rank_of_you(game),
                         "town_p_you": round(game.mind.p_human("reader"), 3)})
        self.last_read = {"p_lie": round(p, 3), "cues": u.cues, "top": top[0] if top else None,
                          "features": u.features, "labelled": is_lie}
        return self.last_read

    def _rank_of_you(self, game: Game) -> int:
        """1 if you are now the town's top suspect, 2 if second, and so on."""
        town = game.mind.public("reader")
        return sorted(town, key=town.get, reverse=True).index(YOU) + 1

    def _on_vote(self, game: Game, transcript: str, raw: dict) -> None:
        candidates = game.living(exclude=YOU)
        target = last_name(transcript, candidates)
        if transcript:
            self._say(YOU, transcript)
        if target is None and game.retries < 1:
            game.retries += 1
            self._say(NARRATOR, "I didn't catch a name. Who do you vote for? " + ", ".join(candidates) + ".")
            self._await("vote", "Say a name: " + ", ".join(candidates))
            return
        game.retries = 0
        target = target or self.rng.choice(candidates)
        mind = game.mind
        votes_reader = mind.votes("reader", target)
        votes_blind = mind.votes("blind", target)
        out = self._tally(votes_reader, target)
        out_blind = self._tally(votes_blind, target)
        self.votes.append(DayVote(game=game.number, day=game.day, you_mafia=game.you_mafia,
                                  votes_on_you_reader=sum(v == YOU for v in votes_reader.values()),
                                  votes_on_you_blind=sum(v == YOU for v in votes_blind.values()),
                                  out_reader=out == YOU, out_blind=out_blind == YOU,
                                  p_you_reader=round(mind.p_human("reader"), 3),
                                  p_you_blind=round(mind.p_human("blind"), 3),
                                  p_catch_reader=round(mind.mafia_caught_prob(votes_reader, target), 3),
                                  p_catch_blind=round(mind.mafia_caught_prob(votes_blind, target), 3)))
        self.belief_log.append({"game": game.number, "day": game.day, "mafia": game.mafia,
                                "you_mafia": game.you_mafia,
                                "beliefs": {b: mind.posterior(b, "reader") for b in mind.villagers()},
                                "beliefs_blind": {b: mind.posterior(b, "blind") for b in mind.villagers()}})
        for b, v in votes_reader.items():
            self._say(b, f"I vote {'you' if v == YOU else v}.")
        game.alive[out] = False
        mind.observe_elimination(out)
        role = "the mafia" if out == game.mafia else "a villager"
        who = "You are" if out == YOU else f"{out} is"
        game.log.append({"type": "vote", "day": game.day, "you_voted": target, "you_voted_mafia": target == game.mafia,
                         "votes": votes_reader, "votes_on_you": sum(v == YOU for v in votes_reader.values()),
                         "out": out, "out_mafia": out == game.mafia, "out_blind": out_blind})
        self._say(NARRATOR, f"{who} eliminated. {'You were' if out == YOU else out + ' was'} {role}.",
                  event={"type": "death", "who": out, "how": "vote", "role": "mafia" if out == game.mafia else "villager"})
        if self._check_over(game):
            return
        if out == YOU:
            self._play_out(game)
            return
        game.phase = "night"
        self._say(NARRATOR, "Night falls. Everyone close your eyes.", event={"type": "night"})
        if game.you_mafia:
            self._say(NARRATOR, "Mafia, open your eyes. Whisper the name of who you want to eliminate.")
            self._await("kill", "Whisper a name: " + ", ".join(game.living(exclude=YOU)))
        else:
            self._night(game, self.rng.choice([p for p in game.living(exclude=game.mafia) if p != YOU]))

    def _on_kill(self, game: Game, transcript: str, raw: dict) -> None:
        candidates = game.living(exclude=YOU)
        target = last_name(transcript, candidates)
        if target is None and game.retries < 1:
            game.retries += 1
            self._say(NARRATOR, "Who? Whisper a name: " + ", ".join(candidates) + ".")
            self._await("kill", "Whisper a name: " + ", ".join(candidates))
            return
        game.retries = 0
        self._night(game, target or self.rng.choice(candidates))

    def _on_again(self, game: Game, transcript: str, raw: dict) -> None:
        if not wants_again(transcript):
            self._await("again", "Say 'play again' for another round")
            return
        self.new_game()

    def _night(self, game: Game, victim: str) -> None:
        game.mind.observe_night_kill(victim)
        game.alive[victim] = False
        game.victim = victim
        game.log.append({"type": "kill", "day": game.day, "victim": victim, "by_you": game.you_mafia})
        game.awaiting = None
        if self._check_over(game):
            return
        game.day += 1
        self._start_day(game)

    def _tally(self, bot_votes: dict[str, str], your_vote: str | None) -> str:
        counts: dict[str, int] = {}
        for v in list(bot_votes.values()) + ([your_vote] if your_vote else []):
            counts[v] = counts.get(v, 0) + 1
        top = max(counts.values())
        tied = sorted(p for p, c in counts.items() if c == top)
        return tied[0] if len(tied) == 1 else random.Random(len(self.votes)).choice(tied)

    def _play_out(self, game: Game) -> None:
        """You were voted out as a villager: the bots finish the game, thinking the same way."""
        mind = game.mind
        while game.phase != "over":
            victim = self.rng.choice(game.living(exclude=game.mafia))
            mind.observe_night_kill(victim)
            game.alive[victim] = False
            self._say(NARRATOR, f"Night falls. {victim} is found dead.", event={"type": "death", "who": victim, "how": "night"})
            if self._check_over(game):
                return
            mind.observe_hunches()
            out = self._tally(mind.votes("reader", None), None)
            game.alive[out] = False
            mind.observe_elimination(out)
            self._say(NARRATOR, f"The town eliminates {out}. {out} was {'the mafia' if out == game.mafia else 'a villager'}.")
            self._check_over(game)

    def _check_over(self, game: Game) -> bool:
        if not game.alive[game.mafia]:
            game.winner = "town"
        elif len(game.living()) <= 2:
            game.winner = "mafia"
        else:
            return False
        game.phase = "over"
        for raw, transcript, llm, is_lie in self.pending_labels:
            self.model.add(raw, transcript, llm, is_lie)
        self.pending_labels = []
        you_won = (game.winner == "mafia") == game.you_mafia
        reveal = "" if game.you_mafia else f" The mafia was {game.mafia}."
        self._say(NARRATOR, f"The {game.winner} wins.{reveal} {'You win!' if you_won else 'You lose.'} "
                            "Say 'play again' for another round.",
                  event={"type": "over", "winner": game.winner, "you_won": you_won, "mafia": game.mafia})
        self._await("again", "Say 'play again' for another round")
        return True

    # ---- views --------------------------------------------------------------------------------
    def view(self) -> dict:
        game = self.current
        lines = self.drain()
        if game is None:
            return {"game": None, "lines": lines, "read": self.last_read}
        over = game.phase == "over"
        players = []
        for p in PLAYERS:
            known = over or not game.alive[p] or p == YOU or (game.you_mafia and p == YOU)
            role = None
            if game.mafia and known:
                role = "mafia" if p == game.mafia else "villager"
            players.append({"name": p, "alive": game.alive[p], "role": role,
                            "card": game.cards.index(p) + 1 if game.cards else None})
        finished = [g for g in self.games if g.phase == "over"]
        return {
            "game": {"number": game.number, "day": game.day, "phase": game.phase,
                     "you_mafia": game.you_mafia if game.mafia else None, "players": players,
                     "awaiting": game.awaiting, "winner": game.winner,
                     "town_p_you": round(game.mind.p_human("reader"), 3) if game.mind else None},
            "lines": lines,
            "read": self.last_read,
            "record": {"games": len(finished), "wins": sum((g.winner == "mafia") == g.you_mafia for g in finished)},
        }
