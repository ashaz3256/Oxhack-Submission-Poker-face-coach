"""The AI town's Bayesian beliefs and voting decisions."""
import math
import random

import pytest

from tilt.beliefs import TownMind

BOTS = ["Ava", "Ben", "Cleo", "Dev", "Eli"]


def mind(mafia="Cleo", seed=1):
    return TownMind(human="You", bots=list(BOTS), mafia=mafia, seed=seed)


def test_starts_uniform_and_posteriors_sum_to_one():
    m = mind()
    post = m.posterior("Ava")
    assert set(post) == {"You", "Ben", "Cleo", "Dev", "Eli"}
    assert all(math.isclose(p, 0.2) for p in post.values())
    m.observe_hunches()
    m.observe_accusation("Dev")
    assert math.isclose(sum(m.posterior("Ben").values()), 1.0)


def test_read_moves_only_the_reader_town_and_is_bayes():
    m = mind()
    m.observe_read(0.8, 0.4, reliability=1.0)
    # prior odds 1:4, likelihood ratio odds(0.8)/odds(0.4) = 6 -> posterior odds 6:4
    assert math.isclose(m.p_human("reader"), 0.6, rel_tol=1e-6)
    assert math.isclose(m.p_human("blind"), 0.2)
    m2 = mind()
    m2.observe_read(0.8, 0.4, reliability=0.0)
    assert math.isclose(m2.p_human("reader"), 0.2)


def test_detector_reliability_needs_a_track_record():
    assert TownMind.detector_reliability([]) == 0.0
    perfect = [(0.9, 1), (0.8, 1), (0.2, 0), (0.1, 0)]
    coin = [(0.5, 1), (0.5, 0), (0.5, 1), (0.5, 0)]
    assert TownMind.detector_reliability(coin) == 0.0
    small = TownMind.detector_reliability(perfect)
    big = TownMind.detector_reliability(perfect * 10)
    assert 0 < small < big < 1


def test_the_dead_are_ruled_out_and_kills_are_evidence():
    m = mind(mafia="Cleo")
    m.observe_night_kill("Ben")
    post = m.posterior("Ava")
    assert "Ben" not in post and math.isclose(sum(post.values()), 1.0)
    m.observe_elimination("Dev")
    assert "Dev" not in m.posterior("Ava")


def test_hunches_point_at_the_mafia_on_average():
    wins = 0
    for seed in range(200):
        m = mind(mafia="Cleo", seed=seed)
        for _ in range(4):
            m.observe_hunches()
        wins += m.top_suspect("Ava", "blind") == "Cleo"
    assert wins / 200 > 0.3  # well above the 1-in-5 of guessing


def test_villagers_vote_where_it_counts_not_just_for_their_favourite():
    m = mind(mafia="Eli")
    # Ava's favourite is Dev (36%) but only one other vote is on Dev; Eli (34%) is tied for the lead.
    post = {"Dev": 0.36, "Eli": 0.34, "You": 0.20, "Ben": 0.05, "Cleo": 0.05}
    others = {"Ava": "Dev", "Ben": "Eli", "Cleo": "Ben", "Dev": "Ben", "Eli": "Dev"}
    # Voting Dev makes a three-way tie (expected catch (.36+.34+.05)/3 = .25); voting Eli puts Eli out (.34).
    assert m._best_vote("Ava", post, others, human_vote="Eli") == "Eli"


def test_a_vote_that_cannot_change_the_outcome_stays_sincere():
    m = mind(mafia="Eli")
    post = {"Dev": 0.36, "Eli": 0.34, "You": 0.20, "Ben": 0.05, "Cleo": 0.05}
    others = {"Ava": "Dev", "Ben": "Eli", "Cleo": "Eli", "Dev": "Eli", "Eli": "Dev"}
    assert m._best_vote("Ava", post, others, human_vote="Eli") == "Dev"


def test_mafia_bot_avoids_its_own_elimination():
    m = mind(mafia="Ben")
    for o in ["Ava", "Cleo", "Dev", "Eli"]:
        m.logw["reader"][o]["Ben"] += 3.0
    votes = m.votes("reader", human_vote="Ava")
    assert votes["Ben"] != "Ben"


def test_game_runs_end_to_end_with_beliefs():
    from tilt.synthetic import play

    s = play(n_games=3, leakiness=1.0, seed=5)
    assert s.votes and all(v.p_you_reader is not None for v in s.votes)
    assert s.belief_log and all(math.isclose(sum(p.values()), 1.0) for row in s.belief_log for p in row["beliefs"].values())


def test_detector_cannot_learn_your_role_within_a_game():
    """Every answer in a game shares one label (your role). If labels were learned as they came, a
    player with no tells at all would be read perfectly by the third answer. They wait for game end."""
    from sklearn.metrics import roc_auc_score

    from tilt.synthetic import play

    later = []
    for i in range(40):
        s = play(n_games=1, leakiness=0.0, seed=2000 + i, same_words=True)
        answers = [u for u in s.utterances if u.kind == "answer" and u.is_lie is not None]
        later += [(u.p_lie, int(u.is_lie)) for u in answers[2:]]
    labels = [y for _, y in later]
    assert 0 < sum(labels) < len(labels)
    assert roc_auc_score(labels, [p for p, _ in later]) < 0.7
