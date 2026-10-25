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

Over the whole file (647,871 frames, 1,568 recordings, 196 people): 59% of frames failed the mesh
check, 99% of clean frames were rotated (262,099 chin-right, 2,351 upright, 669 chin-left), and
only 325 clean frames (0.1%) failed the geometry check. The filter is label-neutral: 39.1% of
frames survive in truthful recordings and 38.4% in deceptive ones. 1,189 recordings from 172
people keep at least 10 clean frames with both a truth and a lie left per person.

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

## Results (`results/face_model_eval.json`)

1,189 recordings, 172 people (581 lies, 608 truths), 5-fold by participant.

| | AUC | Brier |
|---|---|---|
| Face model, split by participant | **0.568** [95% CI 0.525, 0.608] | 0.2481 [0.2454, 0.2507] |
| Same, 5 different fold shuffles | mean 0.556 (0.550–0.568) | |
| Without per-person centring | 0.545 | 0.2490 |
| Random split by recording (not valid) | 0.566 | 0.2490 |
| Chance / constant 0.5 | 0.500 | 0.2500 |

- **It beats chance, modestly.** The whole 95% interval is above 0.5, and shuffling truth/lie within
  each person 200 times never reached it often (p = 0.005; the shuffles averaged 0.499, 95th
  percentile 0.536). The headline 0.568 is the best of five fold shuffles; their mean is 0.556.
- **It knows it is weak.** Strong regularisation was chosen in every fold (C = 0.003), so
  predictions stay between about 0.44 and 0.54, and they are roughly calibrated: the lowest fifth
  of predictions (mean 0.44) were lies 38% of the time, the highest fifth (0.54) 56%.
- **Liars moved less.** Standardised weights: brow movement −0.08, blinking −0.05, head pitch −0.05,
  head yaw −0.05, looking away +0.04, head position +0.03, horizontal gaze +0.02. Less movement under
  deception fits the cognitive-load account in the literature. Every effect is small.
- **Per-person centring matters** (0.568 vs 0.545): what counts is a change from your own normal.
- **The split by recording did not inflate the score here** (0.566), because per-person centring
  already removes most of who someone is. So we do not claim it would have; we split by person
  because that is the question the game asks (a stranger, not someone in the training set).

## In the game

The trained model gives a log-odds score for each spoken answer, centred on the player's own
running mean (`backend/tilt/face_model.py`). It enters the online model two ways: as the starting
prior before the game has seen any of your lies, and as one feature (`i3d_face`) the online model
can reweight for you. Response latency is never part of it, because the dataset has no
question-to-answer timing; latency stays in the online model only.

The trained model is installed (`backend/tilt/pretrained_face_model.json`, from the results
above), and `/api/face-model` reports it. Because its predictions are cautious, it nudges the
first reads and then gives way to what the game learns about you. Without the file the score is 0,
the game behaves exactly as before, and the report says *demonstration mode*.

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
`leakiness`. Dealing is the game's own: you are the mafia half the time, after the game's four-answer
warm-up. 60 sessions × 10 games per row; intervals bootstrap over sessions. **These are simulated players: the numbers test the
reasoning, not how well real people can be read.**

| Simulated player | Caught as mafia: Reader | Blind | Difference (95% CI) | Innocent voted out: Reader | Blind | Brier: Reader | Blind | Guessing |
|---|---|---|---|---|---|---|---|---|
| Placebo (no tells at all) | 23% | 22% | +2 pts [0, +3] | 21% | 21% | 0.174 | 0.175 | 0.173 |
| Leakiness 0.0 | 30% | 17% | +14 pts [+9, +19] | 20% | 19% | 0.167 | 0.175 | 0.172 |
| Leakiness 0.5 | 67% | 15% | +51 pts [+45, +58] | 7% | 10% | 0.131 | 0.169 | 0.168 |
| Leakiness 1.0 | 94% | 9% | +85 pts [+80, +90] | 2% | 8% | 0.103 | 0.168 | 0.166 |
| Leakiness 1.5 | 100% | 9% | +91 pts [+88, +94] | 0% | 6% | 0.086 | 0.165 | 0.166 |

The placebo makes liars and truth-tellers identical in body and words: the Reader town must do
no better than the Blind one: it is within two points (23% vs 22%, and innocents are voted out equally often), and with fair 1-in-6 dealing it is level (17% vs 19%). At
leakiness 0 the simulated liar's words still differ (more hedges and fillers), which is why the
Reader still gains a little there. Files: `results/town_eval_game_deal.json`,
`town_eval_placebo_game_deal.json`, `town_eval_fair_deal.json`, `town_eval_placebo.json`.

Calibration of every villager's beliefs at every vote (leakiness 1):

| Bots said | It was the mafia | Beliefs |
|---|---|---|
| 5% | 2% | 5,796 |
| 14% | 12% | 3,677 |
| 26% | 27% | 2,548 |
| 42% | 48% | 1,095 |
| 59% | 67% | 883 |
| 83% | 93% | 883 |

### A leak we found and fixed

The first version of the placebo only ran with fair dealing and passed. Running it at the game's
real deal exposed a leak: every answer in a game carries the same label (your role), and the lie
model learned each label as soon as you answered. A player with no tells at all was read
perfectly (AUC 1.000) by their third answer, and the town voted them out 46% of the time against
39% for the Blind town. Labels now wait until the game ends, and the town's trust in the detector
only counts earlier games and the warm-up. After the fix that player reads at chance (AUC
0.47–0.55) and the town is level with the Blind one (40% vs 40%). `test_detector_cannot_learn_your_role_within_a_game`
fails if it comes back. Every number in this section was measured after the fix.

Before vs after (`research/compare_bots.py`, the bots on `main` before this work vs this town,
same simulated players; the old bots keep their six-answer warm-up, which only helps them): in the
placebo the old bots voted out an innocent human 42% of the time against 11% for their blind
version, so the old report would tell a perfect poker face that their body gave them away; this
town is 21% vs 21%. With a real tell (leakiness 1) both catch the mafia (99% old, 94% new), and
the human wins less against this town (32% vs 38%).

## Why the warm-up (`research/warmup_experiment.py`)

The same sessions played with 0, 2, 4 or 6 warm-up answers (alternating truth and lie), liars
saying the same words as truth-tellers so only face and voice can give them away. 60 sessions ×
4 games; AUC is the detector's read before it learns each label.

| Warm-up answers | AUC, first game (leakiness 1) | Mafia caught (leakiness 1) | Mafia caught (leakiness 0.5) |
|---|---|---|---|
| 0 | 0.47 [0.39, 0.55] | 42% | 42% |
| 2 | 0.96 [0.93, 0.99] | 83% | 53% |
| 4 | 0.98 [0.97, 1.00] | 90% | 58% |
| 6 | 0.99 [0.97, 1.00] | 98% | 77% |

Without a warm-up the detector cannot read anyone's body, however much they leak: there are no
honest answers to measure "different from your normal" against. Four answers read a strong
leaker almost as well as six (0.98 vs 0.99); six still catch subtler leakers more often (77% vs 58%). The game uses four to
keep the start short.

Known gaps: the bots assume fair dealing, but the game deals you the mafia half the time to
collect lie data; the gut-feeling strength (`HUNCH`) and accusation model (`ACCUSE_HIT`) are
assumptions; and above 40% the bots are underconfident (59% said, 67% actual; 83% said, 93% actual), so they hold back on votes they could make.
