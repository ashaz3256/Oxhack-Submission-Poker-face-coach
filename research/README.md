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
| 2 | `extract_landmark_features.py` | Streams the 18 GB CSV once, drops frames where FaceMesh lost the face, turns the rest upright, writes one row per recording. Never stores the file. `--check` verifies the column mapping on the probe sample with a face-geometry test. |
| 3 | `train_face_model.py` | Regularised logistic regression, evaluated by participant, on recordings with at least 10 clean frames. Writes `results/face_model_eval.json` and, only if the 95% CI on AUC is above 0.5, `backend/tilt/pretrained_face_model.json`. |

Participants: `Video_Name` is `{LIE|TRUTH}_{ethnicity}_{number}_{sex}_..._{take}`, and
`{ethnicity}_{number}_{sex}` identifies a person (196 of them, each with exactly 4 truthful and 4
deceptive recordings in the audio and text files):

    --participant-regex '^(?:LIE|TRUTH)_([A-Za-z]+_\d+_(?:Male|Female))_'

## Data quality: what the landmark file actually contains

Three things about `LandmarkDataset.csv` are not in its documentation. Each would have silently
produced a model trained on the wrong numbers.

1. **Column names are labels, not an index.** The face columns are called `Contour_Chin`,
   `Left_Eye_Upper_0` and so on, but they sit in MediaPipe index order (column *k* is landmark *k*).
   Checked blind: matching the 468 columns of a recording to MediaPipe's canonical face mesh by
   geometry alone (optimal assignment plus affine fit, no names used) put 450 of 468 back in index
   order; the other 18 are near-ties with identical fit. The file also has two metadata columns
   (`ethnic_group`, `ethnic_key`) after `Class`.
2. **The faces are sideways.** In every clean frame sampled, the face is rotated about 90° (forehead
   to chin runs along image x): phone video whose rotation flag was ignored. The extractor turns each
   frame upright before measuring, so yaw, pitch, gaze and blinks are on the axes the game uses.
3. **FaceMesh often lost the face.** MediaPipe returns 468 points even when tracking fails, and on
   sideways faces it often does: the points jump to a distorted shape, with the face angle swinging
   30–40° between frames. A similarity fit to the canonical face separates the two cleanly (clean
   frames 0.05–0.08 relative error, lost frames 0.12–0.18; cut at 0.10). In 25 recordings sampled
   evenly through the file, 9 were mostly clean and 16 mostly lost. After the cut, no frame fails
   the separate face-geometry check (forehead above chin, brows above eyes, and so on).

Dropping frames is only safe if it doesn't depend on truth or lie. The trainer reports the share of
clean frames and the usable recordings for each label (`selection` in `face_model_eval.json`).

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

# How the AI town reasons

`backend/tilt/beliefs.py`. Each villager bot holds a probability that each player is the mafia and
updates it with Bayes' rule from: the lie detector's read on your answer (an odds update,
`logit(p) - logit(base)`, scaled by how much the detector has proven itself on you), silence,
who you accuse, night kills, eliminations, and a weak daily gut feeling about other bots with a
known accuracy (this replaced a hidden bonus that pointed villagers at the bot mafia).

Votes are decisions: each villager casts the vote that maximises the probability that whoever is
eliminated is the mafia, given everyone else's votes (best responses, plurality, random
tie-breaks). The mafia bot votes to minimise its own risk. The Blind town is identical, shares
every random draw, and never hears the lie detector.

**Trust is earned.** A read's weight is the detector's out-of-sample AUC on this player so far,
shrunk toward chance by n/(n+10), mapped to 0..1. Without this, a placebo test (below) showed the
town framing innocent players on pure noise.

## Evidence (simulated players, `research/simulate_town.py`)

The simulated player (`backend/tilt/synthetic.py`) shifts face and voice when lying by
`leakiness`. Dealing is fair here (1 in 6 that you are the mafia, as the bots assume). 60
sessions × 10 games per row; intervals bootstrap over sessions. **These are simulated players:
the numbers test the reasoning, not how well real people can be read.**

| Simulated player | Caught as mafia: Reader | Blind | Difference (95% CI) | Innocent voted out: Reader | Blind | Brier: Reader | Blind | Guessing |
|---|---|---|---|---|---|---|---|---|
| Placebo (no tells at all) | 19% | 21% | -2 pts [-6, +1] | 20% | 20% | 0.174 | 0.173 | 0.172 |
| Leakiness 0.0 | 33% | 20% | +12 pts [+5, +20] | 12% | 14% | 0.167 | 0.172 | 0.171 |
| Leakiness 0.5 | 75% | 13% | +62 pts [+51, +72] | 3% | 10% | 0.142 | 0.169 | 0.170 |
| Leakiness 1.0 | 94% | 11% | +83 pts [+72, +92] | 1% | 7% | 0.131 | 0.169 | 0.169 |
| Leakiness 1.5 | 87% | 12% | +75 pts [+61, +88] | 0% | 7% | 0.132 | 0.170 | 0.169 |

The placebo makes liars and truth-tellers identical in body and words: the Reader town must do
no better than the Blind one, and it doesn't. At leakiness 0 the simulated liar's words still
differ (more hedges and fillers), which is why the Reader still gains a little there.

Calibration of every villager's beliefs at every vote (leakiness 1):

| Bots said | It was the mafia | Beliefs |
|---|---|---|
| 3% | 3% | 5,251 |
| 15% | 18% | 3,437 |
| 26% | 28% | 2,929 |
| 42% | 36% | 1,378 |
| 59% | 50% | 975 |
| 85% | 83% | 745 |

Before vs after (`research/compare_bots.py`, Ismail's bots on `main` vs this town, the game's real 50% deal):
in the placebo the old bots voted out an innocent human 42% of the time against 11% for their blind
version, so the old report would tell a perfect poker face that their body gave them away; the new
town is 26% vs 23%. With a real tell (leakiness 1) both catch the mafia (99% old, 95% new) and the
new town wins more games (the human won 38% against the old bots, 28% against the new).

Known gaps: the bots assume fair dealing, but the game deals you the mafia half the time to
collect lie data; the gut-feeling strength (`HUNCH`) and accusation model (`ACCUSE_HIT`) are
assumptions, and the 50–70% belief band is somewhat overconfident.
