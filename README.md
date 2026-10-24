# Tilt

**Your body is leaking information. Tilt measures how much it costs you.**

You play heads-up poker against a bot while your heart rate and breathing are read from your webcam. One opponent, the **Shark**, reads your vitals live and learns your tells. Another, the **Honest** bot, bets exactly the same way but only knows how often you bluff. Every bet you make is scored against both, so the gap between them is the number of big blinds your body gave away.

Built at OxHack 2026 for the HRT prize, Best Use of Data for Predictions and Decision Making.

## How the prediction works

- **Self-labelling data.** We deal the cards, so every decision is labelled the moment it's made. A bet with under 50% equity against a random hand is a bluff. A call well below the pot odds, or a fold well above them, is a bad decision.
- **Features.** These come from the 10 seconds before each action: heart rate and breathing as z-scores against your first 45 seconds, heart rate slope, and time to act. The tilt model also uses your last hand's result and your drawdown.
- **Shark tell model.** An online logistic regression predicts P(bluff | vitals). It is refit after every bet and shrunk toward your base bluff rate while data is thin.
- **Decision rule.** Both bots call when `p_bluff * equity_vs_weak + (1 - p_bluff) * equity_vs_strong > pot odds`. The Shark gets its `p_bluff` from your body. The Honest bot gets it from your historical frequency.
- **Honest evaluation.**
  - The Shark's AUC uses its read at the time of each bet, before it learns that bet's answer, so it is out-of-sample.
  - The cost of your tells is the paired EV difference on the same bets, assuming a check-down if called. It is reported with a standard error.
- **Tilt model.** This predicts P(bad call or fold) from the same vitals plus tilt context.

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
