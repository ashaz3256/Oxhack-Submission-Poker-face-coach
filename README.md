# Tilt

**Mafia on a video call with five AIs, and every word you say is checked for lies.**

Everyone draws a card, and yours makes you either the **mafia** or a **villager**. Five AI players, each with their own designed voice and portrait, talk, accuse and question you. You never type; you just talk.

While you answer, Tilt watches and listens:
- **Face:** your eyes, eyebrows, mouth and head.
- **Voice:** your pitch, pauses and how fast you speak.
- **Words:** what you say.
- **Heart:** your pulse and breathing, from the webcam.

A lie model learns how *you* leak, and the AI town reasons about it with Bayes' rule. Lie badly and they vote you out.

Built at OxHack 2026 for the HRT prize, Best Use of Data for Predictions and Decision Making, by Ismail (game, interface, Presage) and Arij (data science, AI town, voices).

## The evidence

Every number below comes from the result files in [`research/results/`](research/results/); the method, caveats and how to rerun each one are in [`research/README.md`](research/README.md). The same results as charts: [How Tilt Knows](https://claude.ai/artifact/9XPhtLsqTP3PsanYvXShBn).

| Question | Result |
|---|---|
| Does the face model beat chance on people it never saw? | **AUC 0.568** [95% CI 0.525, 0.608] on 172 people, 1,189 recordings; permutation p = 0.005. Modest, and reported as modest. |
| Do the bots frame an innocent player who has no tells at all? | The original bots did: 42% with the detector vs 11% without. **Tilt's town: 21% vs 21%.** |
| Does the detector help when there is something to read? | With clear tells the mafia is voted out **94% of the time vs 9%** without the detector, and innocents 2% vs 8%. |
| Can you trust the bots' percentages? | Close to calibrated: when they say 26% it is 27%; above 40% they are a little cautious (59% said, 67% actual). |
| Why the warm-up? | With no warm-up the detector reads at chance (AUC 0.47). Four answers: 0.98, almost as good as six (0.99). |

The face model is real people; the town results are simulated players with the same dice for both towns, which is how the reasoning is tested fairly.

**Found and fixed along the way:**
- The published landmark file had three undocumented problems: mislabelled column names, sideways faces, and face tracking that failed on 59% of frames. All three are handled before training, and the cleaning keeps truths and lies alike (39.1% vs 38.4% of frames kept).
- A placebo test exposed a leak: the detector was learning your *role* mid-game. A player with no tells was read perfectly by their third answer. Labels now wait until the game ends, and a test fails if the leak comes back.

## What it measures

| Source | Signals | How |
| --- | --- | --- |
| Face | blink rate, raised brows, frowning, pressed lips, looking away, smile, head movement | MediaPipe Face Landmarker: 478 points plus 52 blendshapes, at about 20 fps in the browser |
| Face (dataset model) | head, gaze, blink and brow movement, scored by a model trained on 172 people | `backend/tilt/face_model.py`, trained in `research/` on the I3D deception dataset (CC BY 4.0) |
| Heart | heart rate, breathing rate | Presage Physiology API when a key is set, otherwise webcam pulse (rPPG) from your forehead |
| Voice | pitch, pitch wobble, loudness, talking speed, pauses, time to start | Web Audio: autocorrelation pitch plus voiced/silent frames |
| Words | hedges ("honestly", "trust me"), ums, denials, answer length | speech-to-text transcript |

Each answer is turned into z-scores against **your own baseline**, taken from the honest answers in a short warm-up at the start.

## How the prediction works

- **Warm-up.** Before the cards, the narrator asks four quick questions and tells you which ones to lie on (two truths, two lies). The truths set your baseline; the lies give the model labelled examples of *your* tells. After each one you see whether it read you right.
- **Model.** A Bayesian logistic regression (MAP, Newton's method) over face, voice, word and language features.
  - Its prior comes from deception research (higher pitch, longer pauses, slower starts, pressed lips, hedges and fillers lean towards a lie) plus the dataset-trained face model. So it reads you from the very first answer instead of waiting for data.
  - Your own answers then pull each weight away from the prior towards how *you* actually leak.
- **Language judge.** Gemini reads each answer in context, without knowing your role, and scores how deceptive the wording is. That score is one more feature, and the model learns how much to trust it for you.
- **Labels in the game.** As the mafia, everything you say in the discussion serves a lie, so every answer of three or more words is labelled a lie when you are the mafia and the truth when you are not. Those labels are only learned **when the game ends**: within a game they all share one label, and learning them as they came would teach the detector your role rather than your tells.
- **Out-of-sample.** Each lie read is made before the model knows that answer's label, so the AUC in the report is prequential and never trained on the answer it scores.
- **Decision.** Each villager bot holds a probability that each player is the mafia and updates it with Bayes' rule from the detector's reads, silences, accusations, night kills and a weak gut feeling. It trusts the detector only as far as the detector has proven itself on *you* in earlier games and the warm-up. Votes are chosen to maximise the chance the mafia is eliminated.
- **Counterfactual.** Every vote is replayed by a blind town that never hears the detector, with the same random draws. The report shows how many days your body got you caught, versus how many it would have been if nobody could read you.

## Run it

On Windows, double-click `start.bat`: it pulls the latest code, starts the backend and frontend in their own windows and opens Chrome. It expects the Python environment at `backend\.venv` (made by the first two backend commands below).

Or by hand:

```bash
# backend
cd backend
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # optional keys: Gemini, ElevenLabs, Presage, Tiger Data
uvicorn tilt.api:app --reload --port 8000

# frontend (second terminal)
cd frontend
npm install
npm run dev                     # open http://localhost:5173 in Chrome or Edge
```

1. Click **Join the call** and allow the camera and microphone.
2. Do the warm-up: answer four questions out loud, lying when the narrator says LIE.
3. After the spooky intro, say a number from one to six, or tap a card.
4. Answer the AIs' questions. At vote time, say a name or tap their tile.
5. When the game ends, read your breakdown: every answer and vote marked as right or wrong, your biggest tells and how to fix them. Say "play again" or tap the button.

**Shortcuts:** Space (or the Done button) sends your answer straight away, → or clicking the captions skips the line being spoken, R replays the last line. After the warm-up a turn ends after your own natural pause length.

**Pause** any time with the button in the sidebar or the Esc key. The current turn is thrown away and asked again when you resume.

**Needs:**
- Chrome or Edge, because they have built-in speech recognition. If it fails (Brave, blocked network, other browsers) the call records each answer and Gemini transcribes it instead, which needs `GEMINI_API_KEY`.
- Headphones, so the AI voices don't leak into your mic.

`?dev=1` gives you a text box instead of the mic, for testing.

## Optional keys (`backend/.env`)

- `GEMINI_API_KEY`: strongly recommended. Gemini judges your words for lies, the AIs improvise their lines, it writes your post-game review, and it transcribes your answers if the browser's speech recognition fails. Without it the lie read uses face, voice and word cues only, and the AIs use templates.
- `ELEVENLABS_API_KEY`: needed for human-sounding voices. Lines use ElevenLabs' `eleven_v4` model with delivery tags (the narrator whispers at night, players gasp, laugh and get sarcastic), falling back to `eleven_multilingual_v2` without tags if v4 isn't available, and are generated ahead so there are no gaps. Each character has a voice designed for them (IDs in `.env.example`); those only work with a key from the ElevenLabs account that made them, and any voice can be overridden with `ELEVENLABS_VOICE_AVA` and so on. The status chip on the call says **Human voices** when it works. If ElevenLabs refuses (bad key, no credits, or a free plan blocked for "unusual activity", common on shared hackathon wifi) the game switches to browser voices and shows ElevenLabs' reason in the sidebar. Open http://localhost:8000/api/voice-check to test the key on its own.
- `PRESAGE_API_KEY`: **Presage** reads your heart rate and breathing rate from the webcam. The call records rolling 15-second clips, sends them to the Presage Physiology API, and the lie model uses Presage's heart rate (instead of our own forehead estimate) plus breathing rate. Readings lag by roughly a clip, and the sidebar shows a Presage chip when it's working. Chrome records MP4 directly; if your browser records WebM, the backend converts it with `ffmpeg` when installed.
- `DATABASE_URL`: a Postgres connection string for **Tiger Data**. Run `backend/schema.sql` once so `utterances` becomes a hypertable. The default is local SQLite.

## Demo safety net

**See a demo report** on the start screen simulates 15 games with a scripted player whose face and voice drift when they lie. The cue sizes in that simulation are made up; real numbers come from real games.

**How accurate is it?** No lie detector is right every time, and nothing reads a perfect poker face: on a simulated player with no tells, the town does no better than chance, by design. On real people the face model alone is modest (AUC 0.568). With clear tells, the town catches the mafia 94% of the time against 9% without the detector. Your report shows your own cross-validated warm-up accuracy and your in-game accuracy.

Tests: `cd backend && pytest -q` (32 tests, including one that fails if the detector can learn your role mid-game).
