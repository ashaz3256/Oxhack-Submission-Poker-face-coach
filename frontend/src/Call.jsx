import { useEffect, useRef, useState } from 'react'
import { api } from './api.js'
import Deal from './Deal.jsx'
import Report from './Report.jsx'
import { FaceScanner } from './sensors/face.js'
import { listen, sttSupported } from './sensors/stt.js'
import { VoiceAnalyser } from './sensors/voice.js'
import { prefetch, say, usingHumanVoices } from './tts.js'

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

  // Live sensor readouts for the side panel, 6 times a second.
  useEffect(() => {
    const id = setInterval(() => {
      setLive({ face: { ...(face.current?.live ?? {}) }, voice: { ...(voice.current?.live ?? {}) } })
    }, 160)
    return () => clearInterval(id)
  }, [])

  useEffect(() => {
    let cancelled = false
    run().catch((e) => setError(e.message))
    return () => {
      cancelled = true
      face.current?.stop()
    }

    async function hear(kind) {
      const measure = kind === 'answer' || kind === 'calibrate'
      if (measure) {
        face.current?.beginTurn()
        voice.current?.beginTurn()
      }
      setListening({ kind, interim: '' })
      let transcript
      if (dev || !sttSupported) {
        transcript = await new Promise((resolve) => (typed.current = resolve))
      } else {
        transcript = await listen({ ...LISTEN[kind], voice: voice.current, onInterim: (t) => setListening({ kind, interim: t }) })
      }
      setListening(null)
      if (!measure) return { transcript, raw: {} }
      const raw = { ...(voice.current?.endTurn(transcript) ?? {}), ...(face.current?.endTurn() ?? {}) }
      return { transcript, raw }
    }

    async function speakLines(lines, state) {
      prefetch(lines)
      for (const line of lines) {
        if (cancelled) return
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
        await sleep(250)
      }
      setSpeaking({ speaker: null, text: '', level: 0 })
    }

    async function run() {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: true })
      videoRef.current.srcObject = stream
      await videoRef.current.play()
      voice.current = new VoiceAnalyser(stream)
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

      prefetch(PRACTICE.map((q) => ({ speaker: 'Narrator', text: q.text })))
      let last = null
      for (const [i, q] of PRACTICE.entries()) {
        setPractice({ round: i + 1, of: PRACTICE.length, lie: q.lie, result: null })
        await speakLines([{ speaker: 'Narrator', text: q.text }], {})
        const { transcript, raw } = await hear('calibrate')
        last = await api.calibrate(sid.current, transcript, raw, q.lie, q.text)
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
      while (!cancelled) {
        setGame(state.game)
        if (state.read) setRead(state.read)
        if (state.game?.awaiting?.kind === 'pick') setDeal({ picked: null, youMafia: null })
        await speakLines(state.lines, state)
        const awaiting = state.game?.awaiting
        if (!awaiting) break
        if (awaiting.kind === 'again') {
          setReportKey((k) => k + 1)
          setShowReport(true)
          const review = await api.review(sid.current).catch(() => null)
          if (review?.text) await speakLines([{ speaker: 'Narrator', text: review.text }], state)
          await speakLines([{ speaker: 'Narrator', text: "Say 'play again' when you're ready." }], state)
        }
        const { transcript, raw } = await hear(awaiting.kind)
        if (awaiting.kind === 'again') setShowReport(false)
        state = await api.speak(sid.current, transcript, raw)
      }
    }
  }, [dev])

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
            return (
              <div key={pl.name} className={`tile bot ${talking ? 'talking' : ''} ${pl.alive ? '' : 'dead'}`}
                style={{ '--lvl': talking ? speaking.level : 0 }}>
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
            <><b className="me">🎙 Your turn</b> {listening.interim || <i>{game?.awaiting?.prompt ?? 'Speak now…'}</i>}</>
          )}
        </div>
        {(dev || !sttSupported) && listening && (
          <form className="devbox" onSubmit={submitDev}>
            <input autoFocus value={devText} onChange={(e) => setDevText(e.target.value)} placeholder="Dev mode: type what you'd say" />
          </form>
        )}
        {error && <p className="error">{error}</p>}
      </div>

      <aside className="detector">
        <h3>Lie detector</h3>
        <div className={`read ${hot ? 'hot' : ''}`}>
          <div className="read-num">{p == null ? '–' : `${Math.round(p * 100)}%`}</div>
          <div className="read-label">chance your last answer was a lie</div>
          <div className="chips">{(read?.cues ?? []).map((c) => <span key={c} className="chip">{c}</span>)}</div>
        </div>
        <div className="sensors">
          <span className={live.face.face ? 'ok' : 'bad'}>Face {live.face.face ? '✓' : '✗'}</span>
          <span className={live.voice.pitch ? 'ok' : 'bad'}>Voice {live.voice.pitch ? '✓' : '…'}</span>
          <span className={live.face.hr ? 'ok' : 'bad'}>Pulse {live.face.hr ? '✓' : '…'}</span>
          <span className={usingHumanVoices() ? 'ok' : 'bad'}>{usingHumanVoices() ? 'Human voices' : 'Browser voices'}</span>
        </div>
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
          <div><b>{live.face.hr ?? '–'}</b><span>heart bpm</span></div>
          <div><b>{read?.features?.rate_z != null ? `${read.features.rate_z > 0 ? '+' : ''}${read.features.rate_z.toFixed(1)}σ` : '–'}</b><span>talking speed</span></div>
        </div>
        <Meter label="Mic level" value={live.voice.level} />
        {!sttSupported && <p className="hint">This browser has no speech recognition. Use Chrome or Edge for hands-free play.</p>}
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
      {deal && <Deal {...deal} />}
      {showReport && sid.current && (
        <div className="report-drawer"><Report sessionId={sid.current} refreshKey={reportKey} /></div>
      )}
    </div>
  )
}

function cardOwners(players) {
  const owners = Array(6).fill(null)
  for (const p of players) if (p.card) owners[p.card - 1] = p.name
  return owners
}
