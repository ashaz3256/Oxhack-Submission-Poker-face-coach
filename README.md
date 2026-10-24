# Tilt

**Mafia on a video call with five AIs, and every word you say is checked for lies.**

Everyone draws a card, and yours makes you either the **mafia** or a **villager**. Five AI players, each with their own voice, talk, accuse and question you. You never type or click; you just talk.

While you answer, Tilt watches and listens:
- **Face:** your eyes, eyebrows and mouth.
- **Voice:** your pitch and how fast you speak.
- **Words:** what you say.

A machine learning model learns when you are lying. The AIs hear your tells, so if you lie badly they vote you out.

Built at OxHack 2026 for the HRT prize, Best Use of Data for Predictions and Decision Making.

## What it measures

| Source | Signals | How |
| --- | --- | --- |
| Face | blink rate, raised brows, frowning, pressed lips, looking away, smile, head movement | MediaPipe Face Landmarker: 478 points plus 52 blendshapes, at about 20 fps in the browser |
| Heart | heart rate | webcam pulse (rPPG) from the green channel of your forehead |
| Voice | pitch, pitch wobble, loudness, talking speed, pauses, time to start | Web Audio: autocorrelation pitch plus voiced/silent frames |
| Words | hedges ("honestly", "trust me"), ums, denials, answer length | speech-to-text transcript |

Each answer is turned into z-scores against **your own baseline**, recorded in two honest calibration questions at the start.

## How the prediction works

- **Labels.**
  - As the mafia, everything you say in the discussion serves a lie. Every answer of three or more words is labelled a lie when you are the mafia and the truth when you are not.
  - The two calibration answers are labelled true.
- **Model.** An online logistic regression over 18 face, voice and word features. It is refit after every answer and shrunk toward your base lie rate while data is thin.
- **Out-of-sample.**
  - Each lie read is made *before* the model learns that answer's label.
  - The AUC in the report is therefore prequential and never trained on the answer it scores.
- **Decision.** The AI town adds `2.5 x (lie read - your base rate)` to its suspicion of you after each answer.
- **Counterfactual.** Every vote is replayed with your tells removed, using the same random draws. The report shows how many days your body got you caught, versus how many it would have been if nobody could read you.

## Run it

```bash
# backend
cd backend
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # optional keys: Gemini, ElevenLabs, Tiger Data
uvicorn tilt.api:app --reload --port 8000

# frontend (second terminal)
cd frontend
npm install
npm run dev                     # open http://localhost:5173 in Chrome or Edge
```

1. Click **Join the call** and allow the camera and microphone. This is the only click.
2. Answer the two calibration questions out loud.
3. When the cards appear, say a number from one to six.
4. Answer the AIs' questions. At vote time, say a name.
5. When the game ends, say "play again".

**Needs:**
- Chrome or Edge, because they have built-in speech recognition. Other browsers get a text box instead.
- Headphones, so the AI voices don't leak into your mic.

`?dev=1` gives you a text box instead of the mic, for testing.

## Optional keys (`backend/.env`)

- `GEMINI_API_KEY`: the AIs improvise their lines and react to your tells, and Gemini writes and speaks your post-game review. Without it they use varied templates.
- `ELEVENLABS_API_KEY`: each player gets their own ElevenLabs voice. Override any voice with `ELEVENLABS_VOICE_AVA` and so on. Without it, each player gets a distinct browser voice with its own pitch and speed.
- `DATABASE_URL`: a Postgres connection string for **Tiger Data**. Run `backend/schema.sql` once so `utterances` becomes a hypertable. The default is local SQLite.

## Demo safety net

**See a demo report** on the start screen simulates 15 games with a scripted player whose face and voice drift when they lie. The cue sizes in that simulation are made up; real numbers come from real games.

Tests: `cd backend && pytest -q`.
