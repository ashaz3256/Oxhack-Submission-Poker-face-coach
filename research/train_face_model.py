"""Step 3: train and evaluate a regularised logistic regression on per-recording face features.

Input: the per-recording CSV written by extract_landmark_features.py
(columns: video, participant, label, then window features).

    python research/train_face_model.py research/data/video_features.csv

Evaluation rules
- Splits are by participant (every recording of a person is on one side), never by frame or video.
- The model only sees FACE_FEATURES. Video names, participant ids and labels are never inputs.
- Regularisation strength is chosen inside each training fold (nested), never on the test fold.
- Uncertainty: bootstrap over participants (cluster bootstrap), plus a permutation test that
  shuffles truth/lie labels within each participant.
- For comparison only, the same model is scored with a naive random split by video, to show how
  much a person appearing on both sides inflates the score.

Writes research/results/face_model_eval.json and, only if the model beats chance with the 95%
interval above 0.5, backend/tilt/pretrained_face_model.json for the game to use.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from tilt.face_features import FACE_FEATURES  # noqa: E402

C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
OUTER_FOLDS = 5
INNER_FOLDS = 4
DOI = "10.5281/zenodo.17421590"


# ---- data ---------------------------------------------------------------------------------------

def load(path: Path, features: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"video", "participant", "label", *features} - set(df.columns)
    if missing:
        raise SystemExit(f"{path} is missing columns: {sorted(missing)}")
    df = df[["video", "participant", "label", *features]].copy()
    if not set(df["label"].unique()) <= {0, 1}:
        raise SystemExit("label must be 0 (truth) / 1 (lie)")
    if df["video"].duplicated().any():
        raise SystemExit("duplicate video rows")
    return df


def centre_within_participant(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Subtract each person's own mean (no labels used). Mirrors Tilt's per-player baseline.

    Missing values are filled with the person's mean first, so a missing feature reads as 'typical
    for this person' (0 after centring), which is also what the game does with a missing window.
    """
    out = df.copy()
    g = out.groupby("participant")[features]
    out[features] = out[features].fillna(g.transform("mean"))
    out[features] = out[features] - out.groupby("participant")[features].transform("mean")
    out[features] = out[features].fillna(0.0)
    return out


def participant_folds(participants: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    ids = np.unique(participants)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    return [np.isin(participants, chunk) for chunk in np.array_split(ids, k)]


# ---- model --------------------------------------------------------------------------------------

def make_model(c: float):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))


def choose_c(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> float:
    best_c, best_ll = C_GRID[0], np.inf
    folds = participant_folds(groups, INNER_FOLDS, seed)
    for c in C_GRID:
        losses = []
        for test in folds:
            m = make_model(c).fit(X[~test], y[~test])
            losses.append(log_loss(y[test], m.predict_proba(X[test])[:, 1], labels=[0, 1]))
        if np.mean(losses) < best_ll:
            best_c, best_ll = c, float(np.mean(losses))
    return best_c


def cross_validated(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int, nested: bool = True):
    """Out-of-fold probabilities with participant-level folds."""
    p = np.full(len(y), np.nan)
    chosen = []
    for i, test in enumerate(participant_folds(groups, OUTER_FOLDS, seed)):
        c = choose_c(X[~test], y[~test], groups[~test], seed + 100 + i) if nested else 0.1
        chosen.append(c)
        p[test] = make_model(c).fit(X[~test], y[~test]).predict_proba(X[test])[:, 1]
    return p, chosen


def video_split(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Naive comparison: random folds by recording, so a person's other recordings sit in training."""
    p = np.full(len(y), np.nan)
    for train, test in KFold(OUTER_FOLDS, shuffle=True, random_state=seed).split(X):
        p[test] = make_model(0.1).fit(X[train], y[train]).predict_proba(X[test])[:, 1]
    return p


# ---- statistics ---------------------------------------------------------------------------------

def scores(y: np.ndarray, p: np.ndarray) -> dict:
    return {"auc": float(roc_auc_score(y, p)), "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0, 1]))}


def cluster_bootstrap(y: np.ndarray, p: np.ndarray, groups: np.ndarray, n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    ids = np.unique(groups)
    rows = {g: np.flatnonzero(groups == g) for g in ids}
    aucs, briers = [], []
    for _ in range(n):
        idx = np.concatenate([rows[g] for g in rng.choice(ids, len(ids), replace=True)])
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
        briers.append(brier_score_loss(y[idx], p[idx]))
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]  # noqa: E731
    return {"auc_95ci": ci(aucs), "brier_95ci": ci(briers), "resamples": len(aucs)}


def permutation_test(X, y, groups, observed_auc: float, n: int, seed: int) -> dict:
    """Shuffle labels within each participant (keeps every person's truth/lie balance), rerun the CV."""
    rng = np.random.default_rng(seed)
    null = []
    for i in range(n):
        y_perm = y.copy()
        for g in np.unique(groups):
            idx = np.flatnonzero(groups == g)
            y_perm[idx] = rng.permutation(y[idx])
        p, _ = cross_validated(X, y_perm, groups, seed + i, nested=False)
        null.append(roc_auc_score(y_perm, p))
    null = np.array(null)
    return {"permutations": n, "null_auc_mean": float(null.mean()), "null_auc_95th": float(np.percentile(null, 95)),
            "p_value": float((1 + np.sum(null >= observed_auc)) / (n + 1))}


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 5) -> list[dict]:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p <= hi)
        if m.any():
            out.append({"p_mean": float(p[m].mean()), "lie_rate": float(y[m].mean()), "n": int(m.sum())})
    return out


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


# ---- main ---------------------------------------------------------------------------------------

def run(path: Path, out_dir: Path, model_out: Path | None, bootstrap: int, permutations: int,
        repeats: int, seed: int = 0) -> dict:
    features = list(FACE_FEATURES)
    raw = load(path, features)
    df = centre_within_participant(raw, features)
    X = df[features].to_numpy(float)
    y = df["label"].to_numpy(int)
    groups = df["participant"].astype(str).to_numpy()
    assert X.shape[1] == len(features)  # only face features go in, nothing derived from names or labels

    per_person = df.groupby("participant")["label"].agg(["count", "sum"])
    counts = {"recordings": int(len(df)), "participants": int(len(per_person)), "lies": int(y.sum()),
              "truths": int(len(y) - y.sum()),
              "participants_with_both_labels": int(((per_person["sum"] > 0) & (per_person["sum"] < per_person["count"])).sum()),
              "missing_feature_fraction": {f: float(raw[f].isna().mean()) for f in features}}

    p_main, chosen_c = cross_validated(X, y, groups, seed)
    main = scores(y, p_main)
    repeat_aucs = [roc_auc_score(y, cross_validated(X, y, groups, seed + 1000 * r)[0]) for r in range(1, repeats)]
    repeat_aucs = [main["auc"], *map(float, repeat_aucs)]

    baselines = {
        "constant_0.5": {"auc": 0.5, "brier": 0.25, "log_loss": float(np.log(2))},
        "base_rate": scores(y, np.full(len(y), y.mean())) | {"auc": 0.5},
    }
    raw_X = raw[features].fillna(raw[features].median()).to_numpy(float)
    uncentred, _ = cross_validated(raw_X, y, groups, seed)
    naive = video_split(X, y, seed)

    result = {
        "dataset": {"doi": DOI, "licence": "CC BY 4.0", "file": path.name},
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "commit": git_commit(),
        "features": features,
        "unit": "one recording (a full answer), features summarised over the whole recording",
        "split": f"{OUTER_FOLDS}-fold by participant, nested {INNER_FOLDS}-fold choice of C",
        "counts": counts,
        "model": main | {"chosen_C_per_fold": chosen_c},
        "auc_over_repeats": {"repeats": len(repeat_aucs), "mean": float(np.mean(repeat_aucs)),
                             "min": float(np.min(repeat_aucs)), "max": float(np.max(repeat_aucs))},
        "bootstrap_by_participant": cluster_bootstrap(y, p_main, groups, bootstrap, seed),
        "permutation_within_participant": permutation_test(X, y, groups, main["auc"], permutations, seed),
        "baselines": baselines,
        "comparisons": {
            "without_per_person_centring": scores(y, uncentred),
            "naive_video_split_NOT_VALID": scores(y, naive),
        },
        "calibration": calibration_table(y, p_main),
    }

    # Final model on all participants, C chosen by participant-level CV.
    c = choose_c(X, y, groups, seed)
    final = make_model(c).fit(X, y)
    scaler, logit = final[0], final[-1]
    beats_chance = result["bootstrap_by_participant"]["auc_95ci"][0] > 0.5
    result["final_model"] = {"C": c, "coefficients_standardised": dict(zip(features, map(float, logit.coef_[0]))),
                             "usable_in_game": bool(beats_chance)}

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "face_model_eval.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame({"video": df["video"], "participant": groups, "label": y, "p_lie_oof": p_main}).to_csv(
        out_dir / "face_model_oof.csv", index=False)

    if model_out is not None:
        if beats_chance:
            model = {
                "features": features,
                "centring": "subtract the player's own running mean of each feature",
                "scaler_mean": list(map(float, scaler.mean_)),
                "scaler_scale": list(map(float, scaler.scale_)),
                "coef": list(map(float, logit.coef_[0])),
                "intercept": float(logit.intercept_[0]),
                "eval": {"auc": main["auc"], "auc_95ci": result["bootstrap_by_participant"]["auc_95ci"],
                         "brier": main["brier"], "participants": counts["participants"],
                         "recordings": counts["recordings"]},
                "source": {"doi": DOI, "licence": "CC BY 4.0", "created": result["created"], "commit": result["commit"]},
            }
            model_out.write_text(json.dumps(model, indent=2))
        elif model_out.exists():
            model_out.unlink()  # never leave a stale model behind that the latest evaluation does not support
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("features_csv", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "research" / "results")
    ap.add_argument("--model-out", type=Path, default=ROOT / "backend" / "tilt" / "pretrained_face_model.json")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--permutations", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=5)
    a = ap.parse_args()
    r = run(a.features_csv, a.out, a.model_out, a.bootstrap, a.permutations, a.repeats)
    m, b, perm = r["model"], r["bootstrap_by_participant"], r["permutation_within_participant"]
    c = r["counts"]
    print(f"{c['recordings']} recordings, {c['participants']} participants ({c['lies']} lies / {c['truths']} truths)")
    print(f"Participant-split AUC {m['auc']:.3f} [95% CI {b['auc_95ci'][0]:.3f}, {b['auc_95ci'][1]:.3f}], "
          f"Brier {m['brier']:.4f} (constant 0.5 gives 0.25)")
    print(f"Permutation p = {perm['p_value']:.3f} ({perm['permutations']} shuffles within participant)")
    print(f"Naive video split (not valid, for comparison): AUC {r['comparisons']['naive_video_split_NOT_VALID']['auc']:.3f}")
    print("Model written for the game." if r["final_model"]["usable_in_game"]
          else "Model NOT written for the game: 95% interval does not exclude chance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
