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

Each answer is turned into z-scores against **your own baseline**, taken from the honest answers in a short warm-up at the start.

## How the prediction works

- **Warm-up.** Before the cards, the narrator asks six quick questions and tells you which ones to lie on (three truths, three lies). These give the model labelled examples of *your* lies, so it knows your tells before the game starts. After each one you see whether it read you right.
- **Labels in the game.** As the mafia, everything you say in the discussion serves a lie. Every answer of three or more words is labelled a lie when you are the mafia and the truth when you are not.
- **Model.** A Bayesian logistic regression (MAP, Newton's method) over face, voice, word and language features.
  - Its prior comes from deception research: higher pitch, longer pauses, slower starts, pressed lips, hedges and fillers lean towards a lie. So it reads you from the very first answer instead of waiting for data.
  - Your own answers then pull each weight away from the prior towards how *you* actually leak.
- **Language judge.** Gemini reads each answer in context, without knowing your role, and scores how deceptive the wording is. That score is one more feature, and the model learns how much to trust it for you.
- **Out-of-sample.**
  - Each lie read is made *before* the model learns that answer's label.
  - The AUC in the report is therefore prequential and never trained on the answer it scores.
- **Decision.** The AI town adds `3.5 x (lie read - your base rate)` to its suspicion of you after each answer.
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
2. Do the warm-up: answer six questions out loud, lying when the narrator says LIE.
3. After the spooky intro, say a number from one to six, or tap a card.
4. Answer the AIs' questions. At vote time, say a name or tap their tile.
5. When the game ends, read your breakdown: every answer and vote marked as right or wrong, your biggest tells and how to fix them. Say "play again" or tap the button.

**Pause** any time with the button in the sidebar or the Esc key. The current turn is thrown away and asked again when you resume.

**Needs:**
- Chrome or Edge, because they have built-in speech recognition. Other browsers get a text box instead.
- Headphones, so the AI voices don't leak into your mic.

`?dev=1` gives you a text box instead of the mic, for testing.

## Optional keys (`backend/.env`)

- `GEMINI_API_KEY`: strongly recommended. Gemini judges your words for lies (a big accuracy boost), the AIs improvise their lines, and it writes your post-game review. Without it the lie read uses face, voice and word cues only, and the AIs use templates.
- `ELEVENLABS_API_KEY`: needed for human-sounding voices. Each player gets their own natural ElevenLabs voice (multilingual v2), and lines are generated ahead so there are no gaps. Override any voice with `ELEVENLABS_VOICE_AVA` and so on. The status chip on the call says **Human voices** when it works. If ElevenLabs refuses (bad key, no credits, or a free plan blocked for "unusual activity", common on shared hackathon wifi) the game switches to browser voices and shows ElevenLabs' reason in the sidebar. Open http://localhost:8000/api/voice-check to test the key on its own. Without it you get the best voices your browser has (Edge's Natural voices sound best).
- `PRESAGE_API_KEY`: **Presage** reads your heart rate and breathing rate from the webcam. The call records rolling 15-second clips, sends them to the Presage Physiology API, and the lie model uses Presage's heart rate (instead of our own forehead estimate) plus breathing rate. Readings lag by roughly a clip, and the sidebar shows a Presage chip when it's working. Chrome records MP4 directly; if your browser records WebM, the backend converts it with `ffmpeg` when installed.
- `DATABASE_URL`: a Postgres connection string for **Tiger Data**. Run `backend/schema.sql` once so `utterances` becomes a hypertable. The default is local SQLite.

## Demo safety net

**See a demo report** on the start screen simulates 15 games with a scripted player whose face and voice drift when they lie. The cue sizes in that simulation are made up; real numbers come from real games.

**How accurate is it?** No lie detector is right every time, and nothing reads a perfect poker face. On simulated players whose cues shift moderately when they lie, the game read is about 70 to 93 percent right, and 96 to 100 percent for players who leak strongly. Your report shows your own cross-validated warm-up accuracy and your in-game accuracy.

Tests: `cd backend && pytest -q`.
