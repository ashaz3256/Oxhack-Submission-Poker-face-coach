"""The dataset-trained face score inside the game. The model weights here are made up for the test."""
import numpy as np

from tilt.cues import FACE_SCORE, LIE_FEATURES
from tilt.face_features import FACE_FEATURES, PRIMITIVES, frame_primitives
from tilt.face_model import FaceScorer
from tilt.models import OnlineLogit
from test_face_research import neutral_face

TEST_MODEL = {  # NOT a trained model: fixed weights so the plumbing can be checked
    "features": FACE_FEATURES, "scaler_mean": [0.0] * len(FACE_FEATURES), "scaler_scale": [0.01] * len(FACE_FEATURES),
    "coef": [0.0] * len(FACE_FEATURES), "intercept": 0.0,
    "eval": {"auc": 0, "auc_95ci": [0, 0], "brier": 0, "participants": 0, "recordings": 0}, "source": {"note": "test"},
}
TEST_MODEL["coef"][FACE_FEATURES.index("gaze_x_sd")] = 1.0


def turn(gaze_jitter: float, n: int = 60, seed: int = 0) -> list[list[float]]:
    rng = np.random.default_rng(seed)
    return [[frame_primitives(neutral_face(gaze=rng.normal(0, gaze_jitter)))[k] for k in PRIMITIVES] for _ in range(n)]


def test_no_model_means_demo_mode_and_zero_score():
    s = FaceScorer(path=__import__("pathlib").Path("/nonexistent.json"))
    assert s.status()["mode"] == "demo"
    assert s.score(turn(0.01))[0] == 0.0


def test_score_is_relative_to_the_players_own_baseline():
    s = FaceScorer(model=TEST_MODEL)
    assert s.score(turn(0.002, seed=1))[0] == 0.0          # first turn has nothing to compare against
    s.score(turn(0.002, seed=2))
    shifty, _ = s.score(turn(0.02, seed=3))                 # much more eye movement than this player's usual
    assert shifty > 0.5
    assert s.score(None) == (0.0, None) and s.score([[None] * len(PRIMITIVES)] * 5)[0] == 0.0


def test_prior_offset_moves_prediction_without_changing_default():
    m = OnlineLogit([], prior_rate=0.4)
    assert m.predict([]) == m.base_rate()
    assert m.predict([], prior_offset=1.0) > m.base_rate() > m.predict([], prior_offset=-1.0)


def test_face_score_flows_into_the_game_and_frames_do_not_leak():
    from tilt.mafia import MafiaSession

    s = MafiaSession(seed=3)
    s.face = FaceScorer(model=TEST_MODEL)
    for i in range(3):
        s.face_frames = turn(0.002, seed=10 + i)
        s.calibrate("I had a quiet day, went to the shops and came home.", {})
        assert s.face_frames is None
    assert FACE_SCORE in LIE_FEATURES
    assert all(u.features[FACE_SCORE] == 0.0 or abs(u.features[FACE_SCORE]) < 4 for u in s.utterances)
    assert s.face.status()["mode"] == "trained"
