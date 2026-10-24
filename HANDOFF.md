# Handoff: Arij's work on Tilt (branch `arij/deception-data`)

Context for whoever picks this up next (including Claude Code). Read this first.

## The goal

OxHack 2026. Main target: **HRT (Hudson River Trading) "Best Use of Data for Predictions and Decision
Making"**. Arij also wants an HRT spring week, so rigour matters more than features: honest
out-of-sample evaluation, calibrated probabilities, confidence intervals, decisions from expected value.
Bonus prizes the project touches: ElevenLabs, Gemini, Tiger Data.

## Team split and rules

- **Ismail** owns the game and UI (`main`) and the **Presage** work (already on `main`). Don't touch Presage.
- **Arij** owns the data science (this branch).
- Commits are authored as **Arij** (ashaz3256). No AI co-author lines.
- **Never push without Arij saying so.** Never push to `main`; merge via pull request.
- Pull `main` and rebase before building on it; Ismail pushes often.
- Run tests from `backend/`: `cd backend && python -m pytest -q` (31 pass).
- This branch is rebased on `main` at `0e1bfdc` (Ismail's Presage / death scenes / breakdown push).

## What this branch adds (on top of Ismail's game)

1. **Dataset face model** (`research/`, `backend/tilt/face_features.py`, `face_model.py`,
   `frontend/src/sensors/faceFrame.js`). Pre-trains face features (head movement, gaze, blinking,
   brows) on the published I3D deception dataset (Zenodo doi:10.5281/zenodo.17421590, CC BY 4.0,
   196 people × 8 testimonies). Participant-split evaluation, bootstrap CIs, permutation test. The
   score enters Ismail's `LieModel` as feature `i3d_face` (prior weight 1.0). Browser and Python
   compute identical features (parity test). **Not trained yet**: no model file ships, so the game
   shows "demonstration mode". No results exist yet; do not invent any.
2. **Expressive voices** (`backend/tilt/voices.py`, `frontend/src/sfx.js`, `backend/scripts/make_sfx.py`).
   Delivery tags via `eleven_v4_turbo`, falling back to Ismail's model; per-character tweaks;
   cache; sound effects and a heartbeat that follows the town's odds on you.
3. **Bayesian AI town** (`backend/tilt/beliefs.py`, `research/simulate_town.py`, `research/compare_bots.py`).
   Bots hold P(mafia) per player, update with Bayes' rule, vote to maximise P(mafia eliminated),
   and trust the lie detector only as much as its out-of-sample AUC on this player has earned.
   Results in `research/README.md` and `research/results/`. Headline: a placebo test showed the
   **old** bots framed innocent players on pure noise (42% vs 11% blind); the new town passes
   (26% vs 23%). Beliefs are well calibrated. All of this is on **simulated** players.

## Open tasks

1. **Train the face model on the real dataset** (needs Arij's laptop; the cloud workspace couldn't reach Zenodo):
   - `python research/fetch_probe.py` → check the schema in `research/data/`
   - Confirm how `Video_Name` maps to a participant, then
     `python research/extract_landmark_features.py --participant-regex '<regex>'` (streams the 18 GB CSV once)
   - `python research/train_face_model.py research/data/video_features.csv`
   - It only writes `backend/tilt/pretrained_face_model.json` if the 95% CI on AUC is above 0.5.
2. **ElevenLabs**: put `ELEVENLABS_API_KEY` in `backend/.env` (never in chat or code), verify with
   `npx -y @elevenlabs/cli@latest onboard check --env-path backend/.env --no-browser`, then
   `cd backend && python scripts/make_sfx.py`. Check the narrator actually whispers at night; if not,
   `eleven_v4_turbo` isn't on the plan and it has fallen back.
3. **Known gaps to consider**: the bots assume fair dealing but the game deals the human the
   mafia 50% of the time; `HUNCH` and `ACCUSE_HIT` in `beliefs.py` are assumptions; the 50–70%
   belief band is a bit overconfident.
4. **Pitch prep**: the placebo story, participant-split vs MSTGNet's video split, and why the
   town's trust in the detector is earned.

## Parked

`arij/presage-parked`: an earlier Presage sidecar, parked because Ismail owns Presage. Only for
reference if he wants any of it.
