# Face model from published deception data

Tilt's live lie model learns each player from scratch. This folder trains a prior for it on
published, labelled data instead of recruiting volunteers or scraping unlabelled video.

**Dataset:** Rahayu, Fatichah, Yuniarti, Probowati & Kusnanti, *Multimodal Deception Detection
Dataset with Personality Profiling across Indonesian Ethnic Groups* (I3D), Zenodo, 2025,
[doi:10.5281/zenodo.17421590](https://doi.org/10.5281/zenodo.17421590). Licence **CC BY 4.0**.
196 participants × 8 recorded testimonies (4 truthful, 4 deceptive) about a theft seen on CCTV.
Only derived features are public: per-frame MediaPipe landmarks (468 face + 10 iris + 33 pose),
audio features and transcripts. No raw video.

Reference implementation: [MSTGNet](https://github.com/yenidwirahayu/MSTGNet) (MIT). Its
splits are by recording (it logs `SUBJECT_ID_UNAVAILABLE` for its subject-overlap audit), so the same
person can be in training and test. We split by participant and report the gap.

## Pipeline

| Step | Script | What it does |
| --- | --- | --- |
| 1 | `fetch_probe.py` | Downloads the small files and the first 3 MB of the 18 GB landmark CSV (HTTP range request) to confirm the schema and how recordings map to people. |
| 2 | `extract_landmark_features.py` | Streams the 18 GB CSV once, keeps 17 landmarks, writes one row per recording. Never stores the file. `--check` verifies the column mapping on the probe sample with a face-geometry test. |
| 3 | `train_face_model.py` | Regularised logistic regression, evaluated by participant. Writes `results/face_model_eval.json` and, only if the 95% CI on AUC is above 0.5, `backend/tilt/pretrained_face_model.json`. |

## Features (`backend/tilt/face_features.py`)

Computed per frame from landmarks, then summarised over a recording (training) or a spoken
answer (game). The browser runs a line-for-line JavaScript copy (`frontend/src/sensors/faceFrame.js`),
checked against the Python by `backend/tests/test_face_parity.py`.

- Head: spread of yaw and pitch proxies (nose vs eyes), spread of nose position in face units
- Gaze: spread of iris position within each eye, share of frames looking away from the usual spot
- Blinking: share of frames with eyes below 60% of usual openness
- Brows: spread of brow height
- Mouth movement is computed but **not used**: dataset participants talk at length, so it mostly measures speech.

Every ratio divides lengths on the same image axis, and every feature is a spread or a share, not
a speed, so camera resolution, aspect ratio and frame rate (iPhone 60 fps vs webcam ~20 fps) cancel out.
Features are centred on each person's own mean, which is how the game uses them too (the player's
running mean of earlier answers).

## Evaluation rules

- Folds by participant; regularisation chosen inside training folds only.
- Inputs are the eight features only: no names, ids, frame numbers or labels.
- AUC and Brier score with 95% intervals from a bootstrap over participants.
- Permutation test that shuffles truth/lie within each participant.
- Baselines: constant 0.5 (AUC 0.5, Brier 0.25). Also shown for comparison: no per-person centring,
  and the invalid random split by recording.

## In the game

The trained model gives a log-odds score for each spoken answer, centred on the player's own
running mean (`backend/tilt/face_model.py`). It enters the online model two ways: as the starting
prior before the game has seen any of your lies, and as one feature (`i3d_face`) the online model
can reweight for you. Response latency is never part of it, because the dataset has no
question-to-answer timing; latency stays in the online model only.

Without a trained model file the score is 0, the game behaves exactly as before, and the report
says *demonstration mode*.

## Known gaps

- Transfer: Indonesian participants, iPhone, describing a video vs English speakers, webcam, playing Mafia.
- The authors report 58–69% accuracy for single modalities, so expect a modest effect, if any.
- Training uses each person's mean over all 8 recordings (4 truth, 4 lie); the game's running mean
  is over however many answers it has seen, with an unknown truth/lie mix.
