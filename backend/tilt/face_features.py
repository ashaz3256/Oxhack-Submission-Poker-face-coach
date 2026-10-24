"""Interpretable face features from MediaPipe landmarks, shared by training and live play.

The same two functions run on the published deception dataset (offline, research/) and on the
player's webcam (live, via frontend/src/faceFrame.js which mirrors `frame_primitives`):

1. `frame_primitives(points)` turns one frame of MediaPipe FaceMesh landmarks (478 points:
   468 face + 10 iris, normalised image coordinates) into a few scale-free numbers.
2. `window_features(frames)` summarises a window of those numbers into FACE_FEATURES.

Every ratio divides by a length on the same image axis (x by x, y by y), so the numbers do not
depend on camera resolution or aspect ratio. Window features are spreads and fractions rather
than speeds, so they do not depend on frame rate either (the dataset and a webcam differ).
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

# MediaPipe FaceMesh indices.
NOSE_TIP = 1
FOREHEAD = 10
CHIN = 152
R_OUTER, R_INNER, R_UPPER, R_LOWER = 33, 133, 159, 145   # subject's right eye (image left)
L_INNER, L_OUTER, L_UPPER, L_LOWER = 362, 263, 386, 374  # subject's left eye
R_BROW, L_BROW = 105, 334
LIP_UPPER, LIP_LOWER = 13, 14
IRIS_A, IRIS_B = 468, 473                                 # iris centres (refined mesh)

USED_LANDMARKS = sorted({NOSE_TIP, FOREHEAD, CHIN, R_OUTER, R_INNER, R_UPPER, R_LOWER, L_INNER, L_OUTER,
                         L_UPPER, L_LOWER, R_BROW, L_BROW, LIP_UPPER, LIP_LOWER, IRIS_A, IRIS_B})

PRIMITIVES = ["yaw", "pitch", "nose_x", "nose_y", "gaze_x", "gaze_y", "eye_open", "brow", "mouth"]

# Window features. Mouth movement is computed for research but kept out of FACE_FEATURES:
# the dataset's participants are talking, a Tilt player is silently picking an answer.
FACE_FEATURES = ["head_yaw_sd", "head_pitch_sd", "head_move_sd", "gaze_x_sd", "gaze_y_sd",
                 "gaze_shift_frac", "blink_frac", "brow_sd"]
RESEARCH_ONLY_FEATURES = ["mouth_sd"]
ALL_WINDOW_FEATURES = FACE_FEATURES + RESEARCH_ONLY_FEATURES

BLINK_RATIO = 0.6      # eye openness below 60% of the window median counts as closed
GAZE_SHIFT = 0.12      # horizontal gaze this far from its median (in eye widths) counts as looking away
MIN_FRAMES = 10

Point = Sequence[float]  # (x, y) or (x, y, z)


def _safe_div(a: float, b: float) -> float:
    return a / b if abs(b) > 1e-9 else float("nan")


def frame_primitives(points: Mapping[int, Point] | Sequence[Point]) -> dict[str, float]:
    """Scale-free measurements for one frame. `points` is indexable by MediaPipe landmark id."""
    p = points
    x = lambda i: float(p[i][0])  # noqa: E731
    y = lambda i: float(p[i][1])  # noqa: E731

    face_h = y(CHIN) - y(FOREHEAD)
    eye_span = x(L_OUTER) - x(R_OUTER)
    mid_x = (x(R_OUTER) + x(L_OUTER)) / 2
    mid_y = (y(R_OUTER) + y(L_OUTER)) / 2

    # Head pose proxies: where the nose tip sits relative to the eyes.
    yaw = _safe_div(x(NOSE_TIP) - mid_x, eye_span)
    pitch = _safe_div(y(NOSE_TIP) - mid_y, face_h)

    # Gaze: iris centre inside each eye, 0 = inner corner, 1 = outer corner, averaged over both eyes.
    # Assign each iris centre to the nearer eye so the result does not rely on iris index order.
    r_centre = (x(R_OUTER) + x(R_INNER)) / 2
    a_is_right = abs(x(IRIS_A) - r_centre) <= abs(x(IRIS_B) - r_centre)
    r_iris, l_iris = (IRIS_A, IRIS_B) if a_is_right else (IRIS_B, IRIS_A)
    gx_r = _safe_div(x(r_iris) - x(R_INNER), x(R_OUTER) - x(R_INNER))
    gx_l = _safe_div(x(l_iris) - x(L_INNER), x(L_OUTER) - x(L_INNER))
    gy_r = _safe_div(y(r_iris) - y(R_UPPER), y(R_LOWER) - y(R_UPPER))
    gy_l = _safe_div(y(l_iris) - y(L_UPPER), y(L_LOWER) - y(L_UPPER))
    # gx_r grows towards the subject's right, gx_l towards their left; flip one so both point the same way.
    gaze_x = (gx_r + (1 - gx_l)) / 2
    gaze_y = (gy_r + gy_l) / 2

    eye_open = _safe_div(((y(R_LOWER) - y(R_UPPER)) + (y(L_LOWER) - y(L_UPPER))) / 2, face_h)
    brow = _safe_div(((y(R_UPPER) - y(R_BROW)) + (y(L_UPPER) - y(L_BROW))) / 2, face_h)
    mouth = _safe_div(y(LIP_LOWER) - y(LIP_UPPER), face_h)

    return {
        "yaw": yaw,
        "pitch": pitch,
        # Nose position in eye-spans / face-heights, so head movement is in face-sized units.
        "nose_x": _safe_div(x(NOSE_TIP), eye_span),
        "nose_y": _safe_div(y(NOSE_TIP), face_h),
        "gaze_x": gaze_x,
        "gaze_y": gaze_y,
        "eye_open": eye_open,
        "brow": brow,
        "mouth": mouth,
    }


def _clean(values: Sequence[float]) -> np.ndarray:
    a = np.asarray(values, dtype=float)
    return a[np.isfinite(a)]


def window_features(frames: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Summarise a window of `frame_primitives` dicts. Returns NaN for every feature if too few frames."""
    if len(frames) < MIN_FRAMES:
        return {k: float("nan") for k in ALL_WINDOW_FEATURES} | {"frames": float(len(frames))}

    col = {k: _clean([f.get(k, float("nan")) for f in frames]) for k in PRIMITIVES}

    def sd(k: str) -> float:
        v = col[k]
        return float(np.std(v)) if len(v) >= MIN_FRAMES else float("nan")

    move = float("nan")
    if len(col["nose_x"]) >= MIN_FRAMES and len(col["nose_y"]) >= MIN_FRAMES:
        move = float(math.hypot(np.std(col["nose_x"]), np.std(col["nose_y"])))

    gx = col["gaze_x"]
    shift = float(np.mean(np.abs(gx - np.median(gx)) > GAZE_SHIFT)) if len(gx) >= MIN_FRAMES else float("nan")

    eo = col["eye_open"]
    blink = float("nan")
    if len(eo) >= MIN_FRAMES and np.median(eo) > 0:
        blink = float(np.mean(eo < BLINK_RATIO * np.median(eo)))

    return {
        "head_yaw_sd": sd("yaw"),
        "head_pitch_sd": sd("pitch"),
        "head_move_sd": move,
        "gaze_x_sd": sd("gaze_x"),
        "gaze_y_sd": sd("gaze_y"),
        "gaze_shift_frac": shift,
        "blink_frac": blink,
        "brow_sd": sd("brow"),
        "mouth_sd": sd("mouth"),
        "frames": float(len(frames)),
    }
