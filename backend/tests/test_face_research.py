"""Checks for the face-feature code and the research training pipeline.

All data here is SYNTHETIC and exists only to test the code: one table with a planted signal
(the pipeline must find it), one with none (the pipeline must not claim one).
"""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tilt.face_features import FACE_FEATURES, frame_primitives, window_features

ROOT = Path(__file__).resolve().parents[2]


def _load_trainer():
    spec = importlib.util.spec_from_file_location("train_face_model", ROOT / "research" / "train_face_model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def neutral_face(dx=0.0, dy=0.0, gaze=0.0, eye=1.0, scale=1.0):
    """A plausible frontal face as 478 normalised points (only the used ones matter)."""
    pts = np.full((478, 3), 0.5)
    s = scale
    def put(i, x, y):
        pts[i] = [0.5 + (x - 0.5) * s + dx, 0.5 + (y - 0.5) * s + dy, 0]
    put(10, 0.5, 0.25); put(152, 0.5, 0.80); put(1, 0.5, 0.55)
    put(33, 0.38, 0.42); put(133, 0.46, 0.42); put(159, 0.42, 0.42 - 0.012 * eye); put(145, 0.42, 0.42 + 0.012 * eye)
    put(362, 0.54, 0.42); put(263, 0.62, 0.42); put(386, 0.58, 0.42 - 0.012 * eye); put(374, 0.58, 0.42 + 0.012 * eye)
    put(105, 0.42, 0.37); put(334, 0.58, 0.37); put(13, 0.5, 0.66); put(14, 0.5, 0.67)
    put(468, 0.42 + gaze, 0.42); put(473, 0.58 + gaze, 0.42)
    return pts


def test_primitives_are_scale_and_translation_free():
    a = frame_primitives(neutral_face())
    b = frame_primitives(neutral_face(dx=0.05, dy=-0.03, scale=1.4))
    for k in ("yaw", "pitch", "gaze_x", "gaze_y", "eye_open", "brow", "mouth"):
        assert math.isclose(a[k], b[k], rel_tol=1e-9, abs_tol=1e-9), k
    assert abs(a["yaw"]) < 1e-9 and math.isclose(a["gaze_x"], 0.5)


def test_gaze_moves_with_iris_and_does_not_depend_on_iris_order():
    left = frame_primitives(neutral_face(gaze=-0.02))["gaze_x"]
    right = frame_primitives(neutral_face(gaze=0.02))["gaze_x"]
    assert left != right
    swapped = neutral_face(gaze=0.02)
    swapped[[468, 473]] = swapped[[473, 468]]
    assert math.isclose(frame_primitives(swapped)["gaze_x"], right)


def test_window_features_blinks_and_short_windows():
    frames = [frame_primitives(neutral_face(eye=0.2 if i % 10 == 0 else 1.0)) for i in range(100)]
    w = window_features(frames)
    assert math.isclose(w["blink_frac"], 0.1)
    assert set(FACE_FEATURES) <= set(w)
    assert math.isnan(window_features(frames[:3])["blink_frac"])


def synthetic_table(signal: float, people: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(people):
        person = rng.normal(0, 1, len(FACE_FEATURES))  # stable individual differences
        for v in range(8):
            label = int(v >= 4)
            x = person + rng.normal(0, 1, len(FACE_FEATURES))
            x[FACE_FEATURES.index("blink_frac")] += signal * label
            rows.append({"video": f"SYN{p:03d}_{v}", "participant": f"SYN{p:03d}", "label": label,
                         **dict(zip(FACE_FEATURES, x))})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("signal,expect_usable", [(1.2, True), (0.0, False)])
def test_training_pipeline_finds_signal_only_when_present(tmp_path, signal, expect_usable):
    trainer = _load_trainer()
    csv = tmp_path / "synthetic_features.csv"
    synthetic_table(signal).to_csv(csv, index=False)
    model_path = tmp_path / "model.json"
    r = trainer.run(csv, tmp_path / "out", model_path, bootstrap=200, permutations=10, repeats=2)
    assert r["counts"]["participants"] == 60 and r["counts"]["participants_with_both_labels"] == 60
    assert r["final_model"]["usable_in_game"] is expect_usable
    assert model_path.exists() is expect_usable
    if expect_usable:
        assert r["model"]["auc"] > 0.7
        assert r["permutation_within_participant"]["p_value"] < 0.2
    else:
        assert r["bootstrap_by_participant"]["auc_95ci"][0] <= 0.5
