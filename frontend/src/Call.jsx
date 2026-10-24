import { useEffect, useRef, useState } from 'react'
import { api } from './api.js'
import Breakdown from './Breakdown.jsx'
import Deal from './Deal.jsx'
import Death from './Death.jsx'
import Intro from './Intro.jsx'
import Pause from './Pause.jsx'
import Report from './Report.jsx'
import { FaceScanner } from './sensors/face.js'
import { PresageRecorder } from './sensors/presage.js'
import { WavRecorder } from './sensors/recorder.js'
import { listen, markBroken, sttState, waitForSpeech } from './sensors/stt.js'
import { VoiceAnalyser } from './sensors/voice.js'
import { playDeath, playIntro, playJoin, playNight, playRing, playVerdict } from './sfx.js'
import { pauseSpeech, prefetch, resumeSpeech, say, voiceStatus } from './tts.js'

const BOTS = ['Ava', 'Ben', 'Cleo', 'Dev', 'Eli']
const FACES = { Ava: '👩‍🦰', Ben: '🧔', Cleo: '👩‍🦱', Dev: '👨‍🦲', Eli: '🧓' }
const LISTEN = {
  calibrate: { maxMs: 15000, silenceMs: 1800 },
  pick: { maxMs: 9000, silenceMs: 900 },
  answer: { maxMs: 22000, silenceMs: 1600 },
  vote: { maxMs: 10000, silenceMs: 1000 },
  kill: { maxMs: 10000, silenceMs: 1000 },
  again: { maxMs: 30000, silenceMs: 1000 },
}
// Practice round: truths set your honest baseline; instructed lies teach the detector what your
// lying looks like, before the game starts.
const PRACTICE = [
  { lie: false, text: "Hey, welcome to Tilt. Quick warm-up so the detector can learn you. First, the truth: tell me about your day so far." },
  { lie: true, text: "Nice. Now lie to me. Tell me what you had for breakfast, but make it up. Sell it." },
  { lie: false, text: "Okay, truth again. What's a film you really like, and why?" },
  { lie: true, text: "Lie this time. Tell me about a country you've never been to, like you've been there." },
  { lie: false, text: "Truth. What did you do last weekend?" },
  { lie: true, text: "Last one, a lie. Tell me about a famous person you've met. You haven't. Convince me." },
]
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
const PARAMS = new URLSearchParams(window.location.search)
const FAST = PARAMS.has('fast') && !PARAMS.has('fx') // automated tests; ?fx keeps full-length animations
const INTRO_MS = FAST ? 300 : 4800
const DEATH_MS = FAST ? 300 : 4600
const PAUSED = '\u0000paused'

function Bars({ level }) {
  return (
    <div className="bars">
      {[0.6, 1, 0.8, 1, 0.6].map((m, i) => (
        <span key={i} style={{ height: `${8 + Math.min(1, level * m * (0.7 + ((i * 37) % 10) / 20)) * 26}px` }} />
      ))}
    </div>
  )
}

function Meter({ label, value, max = 1 }) {
  const pct = value == null ? 0 : Math.max(0, Math.min(1, value / max)) * 100
  return (
    <div className="meter-row">
      <span>{label}</span>
      <div className="bar"><div style={{ width: `${pct}%` }} /></div>
    </div>
  )
}

export default function Call({ dev }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const face = useRef(null)
  const voice = useRef(null)
  const sid = useRef(null)
  const typed = useRef(null)
  const typingRef = useRef(dev)
  const silentFails = useRef(0)
  const recorder = useRef(null)
  const [typing, setTyping] = useState(dev)
  const [status, setStatus] = useState('Starting camera and microphone…')
  const [error, setError] = useState(null)
  const [game, setGame] = useState(null)
  const [speaking, setSpeaking] = useState({ speaker: null, text: '', level: 0 })
  const [listening, setListening] = useState(null)
  const [read, setRead] = useState(null)
  const [deal, setDeal] = useState(null)
  const [live, setLive] = useState({ face: {}, voice: {} })
  const [showReport, setShowReport] = useState(false)
  const [reportKey, setReportKey] = useState(0)
  const [devText, setDevText] = useState('')
  const [practice, setPractice] = useState(null)
  const [paused, setPaused] = useState(false)
  const [intro, setIntro] = useState(null)
  const [death, setDeath] = useState(null)
  const [night, setNight] = useState(false)
  const [verdict, setVerdict] = useState(null)
  const [drawerTab, setDrawerTab] = useState('breakdown')
  const [review, setReview] = useState(null)
  const [joined, setJoined] = useState([])
  const [connecting, setConnecting] = useState(true)
  const presage = useRef(null)
  const pausedRef = useRef(false)
  const waiters = useRef([])
  const choice = useRef(null)      // resolves the current turn from a click
  const interrupt = useRef(null)   // ends the current turn when you pause

  const pause = () => {
    if (pausedRef.current) return
    pausedRef.current = true
    setPaused(true)
    pauseSpeech()
    if (presage.current) presage.current.paused = true
    interrupt.current?.(PAUSED)
  }
  const resume = () => {
    if (!pausedRef.current) return
    pausedRef.current = false
    setPaused(false)
    resumeSpeech()
    if (presage.current) presage.current.paused = false
    waiters.current.splice(0).forEach((r) => r())
  }
  const choose = (text) => choice.current?.(text)

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Escape') (pausedRef.current ? resume : pause)()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  // Live sensor readouts for the side panel, 6 times a second.
  useEffect(() => {
    const id = setInterval(() => {
      setLive({ face: { ...(face.current?.live ?? {}) }, voice: { ...(voice.current?.live ?? {}) }, presage: { ...(presage.current?.live ?? {}) } })
    }, 160)
    return () => clearInterval(id)
  }, [])

  useEffect(() => {
    let cancelled = false
    run().catch((e) => {
      setConnecting(false)
      setError(e.message)
    })
    return () => {
      cancelled = true
      face.current?.stop()
      presage.current?.stop()
    }

    async function gate() {
      if (pausedRef.current) await new Promise((r) => waiters.current.push(r))
    }

    // One turn of you talking. A click (a card, a player) can answer instead, and pausing throws the
    // turn away so it starts fresh, measurements included, when you resume.
    async function hear(kind) {
      const measure = kind === 'answer' || kind === 'calibrate'
      for (;;) {
        await gate()
        if (measure) {
          face.current?.beginTurn()
          voice.current?.beginTurn()
        }
        setListening({ kind, interim: '' })
        const control = {}
        const turnStart = performance.now()
        recorder.current?.start()
        const backup = !dev && sttState.broken
        let heard
        if (dev || typingRef.current) heard = new Promise((resolve) => (typed.current = resolve))
        else if (backup) heard = waitForSpeech({ ...LISTEN[kind], voice: voice.current, control })
        else heard = listen({ ...LISTEN[kind], voice: voice.current, control, onInterim: (t) => setListening({ kind, interim: t }) })
        const clicked = new Promise((resolve) => (choice.current = resolve))
        const stopped = new Promise((resolve) => (interrupt.current = resolve))
        const [source, said] = await Promise.race([
          heard.then((t) => ['speech', t]), clicked.then((t) => ['click', t]), stopped.then((t) => ['pause', t])])
        choice.current = null
        interrupt.current = null
        control.stop?.('')
        const audio = recorder.current?.stop()
        if (source === 'pause') {
          setListening(null)
          if (measure) {
            voice.current?.endTurn('')
            face.current?.endTurn()
          }
          continue
        }
        const faceRaw = measure ? face.current?.endTurn() ?? {} : {}
        let transcript = said ?? ''
        // The browser heard nothing but the mic did: its recognition isn't working here, so send
        // the recording to the backup transcriber instead (and use it from now on after twice).
        const spoke = voice.current && voice.current.lastVoiceAt > turnStart + 200
        if (source === 'speech' && !dev && !typingRef.current && audio && (backup || (!transcript && spoke))) {
          if (!backup && ++silentFails.current >= 2) markBroken('Browser speech recognition kept hearing nothing')
          setListening({ kind, interim: '…transcribing what you said' })
          try {
            transcript = await api.transcribe(audio)
          } catch (e) {
            markBroken(`Speech-to-text failed (${e.message}). Type your answers instead.`)
            typingRef.current = true
            setTyping(true)
            setListening({ kind, interim: '' })
            transcript = await new Promise((resolve) => (typed.current = resolve))
          }
        } else if (transcript) {
          silentFails.current = 0
        }
        setListening(null)
        if (!measure) return { transcript, raw: {}, faceFrames: null }
        const raw = { ...(voice.current?.endTurn(transcript) ?? {}), ...faceRaw }
        return { transcript, raw, faceFrames: face.current?.lastFrames ?? null }
      }
    }

    async function speakLines(lines, state) {
      prefetch(lines)
      for (const line of lines) {
        if (cancelled) return
        await gate()
        const ev = line.event
        const started = performance.now()
        if (ev?.type === 'intro') {
          setDeal(null)
          setNight(false)
          setIntro(ev.game)
          playIntro()
          await sleep(INTRO_MS)
          setIntro(null)
          setDeal({ picked: null, youMafia: null })
        }
        if (ev?.type === 'night') {
          setNight(true)
          playNight()
        }
        if (ev?.type === 'death') {
          setDeath(ev)
          playDeath()
        }
        if (ev?.type === 'over') {
          setNight(false)
          setVerdict({ won: ev.you_won, mafia: ev.mafia })
          playVerdict(ev.you_won)
        }
        setSpeaking({ speaker: line.speaker, text: line.text, level: 0 })
        const drew = line.text.match(/^You drew card (\d)/)
        if (drew && state.game) {
          setDeal({ picked: Number(drew[1]), youMafia: state.game.you_mafia, cards: cardOwners(state.game.players) })
        }
        await say(line.speaker, line.text, (level) => setSpeaking((s) => ({ ...s, level })))
        if (drew) {
          await sleep(1200)
          setDeal(null)
        }
        if (ev?.type === 'death') {
          await sleep(Math.max(0, DEATH_MS - (performance.now() - started)))
          setDeath(null)
          if (ev.how === 'night') setNight(false)
        }
        if (ev?.type === 'over') {
          await sleep(FAST ? 100 : 1500)
          setVerdict(null)
        }
        await sleep(250)
      }
      setSpeaking({ speaker: null, text: '', level: 0 })
    }

    async function run() {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: true })
      videoRef.current.srcObject = stream
      await videoRef.current.play()
      voice.current = new VoiceAnalyser(stream)
      recorder.current = new WavRecorder(stream)
      setStatus('Loading the face scanner…')
      try {
        face.current = new FaceScanner()
        await face.current.init()
        face.current.start(videoRef.current, canvasRef.current)
      } catch (e) {
        face.current = null
        setError(`Face scanner failed to load (${e.message}). Voice and words are still analysed.`)
      }
      setStatus(null)
      sid.current = (await api.create()).session_id
      presage.current = new PresageRecorder(stream, sid.current)
      presage.current.start()

      // Ring out, then the five AIs join the call one by one.
      playRing()
      await sleep(FAST ? 50 : 2200)
      for (const [i, name] of BOTS.entries()) {
        setJoined((j) => [...j, name])
        playJoin(i)
        await sleep(FAST ? 20 : 420)
      }
      await sleep(FAST ? 20 : 700)
      setConnecting(false)

      prefetch(PRACTICE.map((q) => ({ speaker: 'Narrator', text: q.text })))
      let last = null
      for (const [i, q] of PRACTICE.entries()) {
        setPractice({ round: i + 1, of: PRACTICE.length, lie: q.lie, result: null })
        await speakLines([{ speaker: 'Narrator', text: q.text }], {})
        const { transcript, raw, faceFrames } = await hear('calibrate')
        last = await api.calibrate(sid.current, transcript, raw, q.lie, q.text, faceFrames)
        setRead({ p_lie: last.p_lie, cues: last.cues })
        setPractice({ round: i + 1, of: PRACTICE.length, lie: q.lie, result: last })
        await sleep(1600)
      }
      setPractice(null)
      const acc = last?.accuracy
      const summary = acc != null
        ? `Done. On your practice answers the detector is ${Math.round(acc * 100)} percent accurate, and it'll keep learning you as we play. Right, let's play.`
        : "Done. The detector's got a feel for you now. Let's play."
      await speakLines([{ speaker: 'Narrator', text: summary }], {})

      let state = await api.newGame(sid.current)
      let reviewed = null
      while (!cancelled) {
        setGame(state.game)
        if (state.read) setRead(state.read)
        const intro = state.lines.some((l) => l.event?.type === 'intro')
        if (state.game?.awaiting?.kind === 'pick' && !intro) setDeal({ picked: null, youMafia: null })
        await speakLines(state.lines, state)
        const awaiting = state.game?.awaiting
        if (!awaiting) break
        if (awaiting.kind === 'again' && reviewed !== state.game.number) {
          reviewed = state.game.number
          setReview(null)
          setDrawerTab('breakdown')
          setReportKey((k) => k + 1)
          setShowReport(true)
          const r = await api.review(sid.current).catch(() => null)
          if (r?.text) {
            setReview(r.text)
            await speakLines([{ speaker: 'Narrator', text: r.text }], state)
          }
          await speakLines([{ speaker: 'Narrator', text: "Have a look at your breakdown. Say 'play again' when you're ready." }], state)
        }
        const { transcript, raw, faceFrames } = await hear(awaiting.kind)
        state = await api.speak(sid.current, transcript, raw, faceFrames)
        if (state.game?.awaiting?.kind !== 'again') setShowReport(false)
      }
    }
  }, [dev])

  const voice_ = voiceStatus()
  const players = game?.players ?? [{ name: 'You', alive: true }, ...BOTS.map((name) => ({ name, alive: true }))]
  const you = players.find((p) => p.name === 'You')
  const p = read?.p_lie
  const hot = p != null && p > 0.6

  const submitDev = (e) => {
    e.preventDefault()
    typed.current?.(devText)
    typed.current = null
    setDevText('')
  }

  return (
    <div className="call">
      <div className="stage">
        <div className="grid">
          <div className={`tile you ${listening ? 'talking' : ''}`}>
            <video ref={videoRef} muted playsInline />
            <canvas ref={canvasRef} />
            <div className="scanline" />
            <div className="tile-label">
              You {you?.role && <span className={`role ${you.role}`}>{you.role}</span>}
              {listening && <Bars level={live.voice.level ?? 0} />}
            </div>
            {!live.face.face && !status && <div className="no-face">Looking for your face…</div>}
          </div>
          {players.filter((pl) => pl.name !== 'You').map((pl) => {
            const talking = speaking.speaker === pl.name
            const pickable = pl.alive && (listening?.kind === 'vote' || listening?.kind === 'kill')
            return (
              <div key={pl.name} className={`tile bot ${talking ? 'talking' : ''} ${pl.alive ? '' : 'dead'} ${pickable ? 'pickable' : ''} ${joined.includes(pl.name) ? 'in' : 'out'}`}
                style={{ '--lvl': talking ? speaking.level : 0 }}
                onClick={pickable ? () => choose(listening.kind === 'vote' ? `I vote ${pl.name}` : pl.name) : undefined}>
                {pickable && <div className="pick-hint">{listening.kind === 'vote' ? 'Vote' : 'Kill'}</div>}
                <div className="avatar-ring"><span className="avatar">{FACES[pl.name]}</span></div>
                <div className="tile-label">
                  {pl.name}
                  {pl.card && <span className="card-chip">card {pl.card}</span>}
                  {pl.role && <span className={`role ${pl.role}`}>{pl.alive ? pl.role : `✝ ${pl.role}`}</span>}
                  {talking && <Bars level={speaking.level} />}
                </div>
              </div>
            )
          })}
        </div>

        <div className={`captions ${speaking.speaker === 'Narrator' ? 'narrator' : ''}`}>
          {status && <span>{status}</span>}
          {!status && speaking.speaker && <><b>{speaking.speaker}</b> {speaking.text}</>}
          {!status && !speaking.speaker && listening && (
            <><b className="me">🎙 Your turn</b> {listening.interim || <i>{game?.awaiting?.prompt ?? 'Speak now…'}{(listening.kind === 'vote' || listening.kind === 'kill') ? ', or tap their tile' : ''}</i>}</>
          )}
        </div>
        {typing && listening && (
          <form className="devbox" onSubmit={submitDev}>
            <input autoFocus value={devText} onChange={(e) => setDevText(e.target.value)} placeholder="Dev mode: type what you'd say" />
          </form>
        )}
        {error && <p className="error">{error}</p>}
      </div>

      <aside className="detector">
        <div className="det-head">
          <h3>Lie detector</h3>
          <button className="pause-btn" onClick={pause} title="Pause (Esc)">⏸ Pause</button>
        </div>
        <div className={`read ${hot ? 'hot' : ''}`}>
          <div className="read-num">{p == null ? '–' : `${Math.round(p * 100)}%`}</div>
          <div className="read-label">chance your last answer was a lie</div>
          <div className="chips">{(read?.cues ?? []).map((c) => <span key={c} className="chip">{c}</span>)}</div>
        </div>
        <div className="sensors">
          <span className={live.face.face ? 'ok' : 'bad'}>Face {live.face.face ? '✓' : '✗'}</span>
          <span className={live.voice.pitch ? 'ok' : 'bad'}>Voice {live.voice.pitch ? '✓' : '…'}</span>
          {!dev && <span className={sttState.broken ? 'bad' : 'ok'} title={sttState.error ?? ''}>{typing ? 'Typing' : sttState.broken ? 'Speech (Gemini)' : 'Speech ✓'}</span>}
          <span className={live.face.hr ? 'ok' : 'bad'}>Pulse {live.face.hr ? '✓' : '…'}</span>
          {live.presage?.enabled && (
            <span className={live.presage.status === 'ok' ? 'ok' : 'bad'} title={live.presage.error ?? 'Heart and breathing rate from Presage'}>
              Presage {live.presage.status === 'ok' ? '✓' : live.presage.status === 'error' ? '✗' : '…'}
            </span>
          )}
          <span className={voice_.human ? 'ok' : 'bad'} title={voice_.error ?? ''}>{!voice_.checked ? 'Voices …' : voice_.human ? 'Human voices ✓' : 'Browser voices'}</span>
        </div>
        {!voice_.human && voice_.keySet && voice_.error && <p className="voice-warn">Human voices off: {voice_.error}</p>}
        <h4>Face</h4>
        <Meter label="Blinking" value={live.face.blink} />
        <Meter label="Brows up" value={live.face.brow_raise} max={0.6} />
        <Meter label="Frowning" value={live.face.brow_down} max={0.6} />
        <Meter label="Lips pressed" value={live.face.lip_press} max={0.5} />
        <Meter label="Looking away" value={live.face.gaze_away} max={0.7} />
        <Meter label="Smile" value={live.face.smile} />
        <h4>Voice and body</h4>
        <div className="stats">
          <div><b>{live.voice.pitch ?? '–'}</b><span>pitch Hz</span></div>
          <div><b>{live.presage?.hr ?? live.face.hr ?? '–'}</b><span>heart bpm{live.presage?.hr ? ' (Presage)' : ''}</span></div>
          <div><b>{live.presage?.rr ?? '–'}</b><span>breaths/min</span></div>
          <div><b>{read?.features?.rate_z != null ? `${read.features.rate_z > 0 ? '+' : ''}${read.features.rate_z.toFixed(1)}σ` : '–'}</b><span>talking speed</span></div>
        </div>
        <Meter label="Mic level" value={live.voice.level} />
        {sttState.broken && !dev && <p className="voice-warn">{sttState.error}. {typing ? '' : 'Using Gemini to transcribe instead, so answers take a couple of seconds longer.'}</p>}
        {game && <p className="hint">Game {game.number} · Day {game.day}{game.you_mafia != null ? ` · you are ${game.you_mafia ? 'the MAFIA' : 'a villager'}` : ''}</p>}
      </aside>

      {practice && (
        <div className={`practice ${practice.lie ? 'lie' : 'truth'}`}>
          <div className="practice-step">Warm-up {practice.round} of {practice.of}</div>
          <div className="practice-ask">{practice.lie ? 'Tell a LIE' : 'Tell the TRUTH'}</div>
          {practice.result && (
            <div className="practice-result">
              Detector: {Math.round(practice.result.p_lie * 100)}% lie{' '}
              {(practice.result.p_lie > 0.5) === practice.lie ? '✓ read you right' : '✗ fooled it'}
            </div>
          )}
        </div>
      )}
      {connecting && (
        <div className="connecting">
          <div className="ring-pulse"><span>📞</span></div>
          <div className="connecting-text">{joined.length ? 'Players joining…' : status ?? 'Calling the town…'}</div>
          <div className="joins">
            {joined.map((n) => <div key={n} className="join-toast">{FACES[n]} <b>{n}</b> joined the call</div>)}
          </div>
        </div>
      )}
      {night && !death && <div className="night-veil"><div className="night-moon" /></div>}
      {deal && <Deal {...deal} onPick={listening?.kind === 'pick' ? (n) => choose(`card ${n}`) : null} />}
      {intro && <Intro game={intro} />}
      {death && <Death {...death} />}
      {verdict && (
        <div className={`verdict ${verdict.won ? 'won' : 'lost'}`}>
          <div className="verdict-text">{verdict.won ? 'YOU WIN' : 'YOU LOSE'}</div>
        </div>
      )}
      {showReport && sid.current && (
        <div className="report-drawer">
          <div className="drawer-bar">
            <div className="tabs">
              <button className={drawerTab === 'breakdown' ? 'on' : ''} onClick={() => setDrawerTab('breakdown')}>Breakdown</button>
              <button className={drawerTab === 'report' ? 'on' : ''} onClick={() => setDrawerTab('report')}>Detector stats</button>
            </div>
            <button className="primary" onClick={() => choose('play again')}>Play again</button>
          </div>
          {drawerTab === 'breakdown'
            ? <Breakdown sessionId={sid.current} refreshKey={reportKey} review={review} />
            : <Report sessionId={sid.current} refreshKey={reportKey} />}
        </div>
      )}
      {paused && <Pause onResume={resume} />}
    </div>
  )
}

function cardOwners(players) {
  const owners = Array(6).fill(null)
  for (const p of players) if (p.card) owners[p.card - 1] = p.name
  return owners
}
