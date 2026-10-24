# Tilt

**Your body is leaking information. Tilt measures how much it costs you.**

You play **Mafia** against five AI players while your heart rate and breathing are read from your webcam. Every day an **Inspector** questions you. Because we know your secret role, every answer is labelled as a lie or the truth the moment you give it.

There are two Inspectors, and they share every rule but one:
- The **Reader** checks your vitals while you answer.
- The **Blind** Inspector only knows how often you lie.

Every day's vote is computed both ways with the same dice. The extra votes your body drew, and the days it got you caught, are measured directly.

A poker mode is also included. In it a **Shark** bot reads your vitals to call your bluffs, and an **Honest** bot does not.

Built at OxHack 2026 for the HRT prize, Best Use of Data for Predictions and Decision Making.

## How the prediction works (poker mode)

- **Self-labelling data.** We deal the cards, so every decision is labelled the moment it's made. A bet with under 50% equity against a random hand is a bluff. A call well below the pot odds, or a fold well above them, is a bad decision.
- **Features.** These come from the 10 seconds before each action: heart rate and breathing as z-scores against your first 45 seconds, heart rate slope, and time to act. The tilt model also uses your last hand's result and your drawdown.
- **Shark tell model.** An online logistic regression predicts P(bluff | vitals). It is refit after every bet and shrunk toward your base bluff rate while data is thin.
- **Decision rule.** Both bots call when `p_bluff * equity_vs_weak + (1 - p_bluff) * equity_vs_strong > pot odds`. The Shark gets its `p_bluff` from your body. The Honest bot gets it from your historical frequency.
- **Honest evaluation.**
  - The Shark's AUC uses its read at the time of each bet, before it learns that bet's answer, so it is out-of-sample.
  - The cost of your tells is the paired EV difference on the same bets, assuming a check-down if called. It is reported with a standard error.
- **Tilt model.** This predicts P(bad call or fold) from the same vitals plus tilt context.

## How the prediction works (Mafia)

- **Labels.** Some answers are lies when you are the mafia: denying it, accusing someone, or claiming you were asleep. Claiming you saw someone at night is a lie for anyone. Evasions are not labelled, but they raise suspicion.
- **Features.** The same vitals features as poker, measured over the 10 seconds before you answer, plus the time you took to answer.
- **The Inspector's lie model.** An online logistic regression that is refit after every answer. The town adds `2.5 x (today's mean lie read - your base lie rate)` to its suspicion of you.
- **Metrics.**
  - Out-of-sample lie-detection AUC.
  - Votes on you per day as the mafia, Reader vs Blind.
  - Days you were caught, Reader vs Blind.
- **Simulated check.** With simulated vitals (leakiness 1), the Reader caught the mafia on 7 to 9 days, against 1 to 4 for the Blind Inspector. That is over 20 games with 8 to 12 as the mafia. With leakiness 0 the two were about equal.

## Run it

```bash
# backend
cd backend
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # optional keys: Tiger Data, Gemini, ElevenLabs
uvicorn tilt.api:app --reload --port 8000

# frontend (second terminal)
cd frontend
npm install
npm run dev                     # http://localhost:5173
```

Tests: `cd backend && pytest -q`.

## Vitals sources

| Source | What it is |
| --- | --- |
| Simulated | Arousal rises with weak hands in big pots and after big losses. Use the leakiness slider to set how much shows; 0 is a perfect poker face. Use this mode when the camera or the SDK is unavailable. |
| Webcam pulse | In-browser rPPG: the green channel of a forehead patch, then the FFT peak in the 45–180 bpm band. Needs steady light. |
| Presage | Pipe SmartSpectra readings as JSON lines into `backend/presage_bridge.py --session <id>`. |

## Stack

- React (Vite) front end
- FastAPI back end
- scikit-learn for the models
- `treys` for hand evaluation
- Postgres on **Tiger Data**: run `backend/schema.sql` once so `vitals` becomes a hypertable
- **Gemini** for the session review
- **ElevenLabs** for the voice coach (falls back to the browser's voice)

## Demo mode

**Demo report (simulated)** on the start screen plays 60 hands with a scripted player and simulated vitals. You get a full leak report even if the live setup fails on stage.
