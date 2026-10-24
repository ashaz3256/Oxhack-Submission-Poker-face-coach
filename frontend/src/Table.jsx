import { useCallback, useEffect, useRef, useState } from 'react'
import { api, speak } from './api.js'
import { useWebcamPulse } from './useWebcamPulse.js'

const SUITS = { s: '♠', h: '♥', d: '♦', c: '♣' }
const ACTION_LABELS = { check: 'Check', bet_half: 'Bet ½ pot', bet_pot: 'Bet pot', call: 'Call', fold: 'Fold' }

export function Card({ card, hidden }) {
  if (hidden || !card) return <div className="card back" />
  const rank = card[0] === 'T' ? '10' : card[0]
  const suit = card[1]
  return (
    <div className={`card ${suit === 'h' || suit === 'd' ? 'red' : ''}`}>
      <span>{rank}</span>
      <span>{SUITS[suit]}</span>
    </div>
  )
}

function Sparkline({ values, width = 260, height = 60 }) {
  if (values.length < 2) return <svg width={width} height={height} />
  const lo = Math.min(...values) - 2
  const hi = Math.max(...values) + 2
  const pts = values.map((v, i) => `${(i / (values.length - 1)) * width},${height - ((v - lo) / (hi - lo)) * height}`)
  return (
    <svg width={width} height={height} className="spark">
      <polyline points={pts.join(' ')} fill="none" strokeWidth="2" />
    </svg>
  )
}

export default function Table({ session, onReport }) {
  const sid = session.id
  const [state, setState] = useState(null)
  const [live, setLive] = useState(null)
  const [hrs, setHrs] = useState([])
  const [voice, setVoice] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const lastLine = useRef('')
  const pending = useRef([])

  const onSample = useCallback((s) => {
    pending.current.push(s)
    setHrs((h) => [...h.slice(-90), s.hr])
  }, [])
  const cam = useWebcamPulse(session.vitals_source === 'webcam', onSample)

  // Vitals loop: simulated ticks, or flush webcam samples to the backend.
  useEffect(() => {
    const id = setInterval(async () => {
      try {
        if (session.vitals_source === 'sim') {
          const s = await api.simTick(sid)
          setHrs((h) => [...h.slice(-90), s.hr])
        } else if (pending.current.length) {
          const batch = pending.current
          pending.current = []
          await api.sendVitals(sid, batch)
        }
        setLive(await api.live(sid))
      } catch {
        // keep ticking; a dropped sample is fine
      }
    }, 1000)
    return () => clearInterval(id)
  }, [sid, session.vitals_source])

  useEffect(() => {
    api.newHand(sid).then(setState).catch((e) => setError(e.message))
  }, [sid])

  useEffect(() => {
    if (!voice || !live?.line || live.line === lastLine.current) return
    lastLine.current = live.line
    if (live.line.startsWith("You're broadcasting") || live.line.startsWith('Heart rate is up')) speak(live.line)
  }, [live, voice])

  const act = async (action) => {
    setBusy(true)
    setError(null)
    try {
      const res = await api.act(sid, action)
      setState(res)
      if (res.live) setLive(res.live)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  const next = async () => setState(await api.newHand(sid))

  const hand = state?.hand
  const done = hand?.phase === 'done'
  const read = live?.shark_p_bluff ?? 0

  return (
    <div className="table-layout">
      <section className="felt panel">
        <div className="seat">
          <div className="who">{session.opponent === 'shark' ? 'Shark' : 'Honest bot'}</div>
          <div className="cards">
            <Card card={hand?.bot?.[0]} hidden={!hand?.bot} />
            <Card card={hand?.bot?.[1]} hidden={!hand?.bot} />
          </div>
        </div>
        <div className="board">
          {[0, 1, 2, 3, 4].map((i) => <Card key={i} card={hand?.board?.[i]} hidden={!hand?.board?.[i]} />)}
        </div>
        <div className="pot">Pot {hand?.pot ?? 0} bb {hand?.to_call ? `· to call ${hand.to_call}` : ''}</div>
        <div className="seat">
          <div className="cards">
            <Card card={hand?.hero?.[0]} />
            <Card card={hand?.hero?.[1]} />
          </div>
          <div className="who">You · {state?.net_bb ?? 0} bb over {state?.hands_played ?? 0} hands</div>
        </div>
        <div className="actions">
          {!done && hand?.legal.map((a) => (
            <button key={a} disabled={busy} className={a.startsWith('bet') ? 'primary' : ''} onClick={() => act(a)}>
              {ACTION_LABELS[a]}
            </button>
          ))}
          {done && (
            <>
              <span className={`result ${hand.result_bb >= 0 ? 'win' : 'loss'}`}>
                {hand.result_bb >= 0 ? '+' : ''}{hand.result_bb} bb
              </span>
              <button className="primary" onClick={next}>Next hand</button>
              <button onClick={onReport}>Leak report</button>
            </>
          )}
        </div>
        <ol className="log">{hand?.log.map((l, i) => <li key={i}>{l}</li>)}</ol>
        {error && <p className="error">{error}</p>}
      </section>

      <aside className="panel side">
        <h3>What the Shark sees</h3>
        {session.vitals_source === 'webcam' && (
          <div className="cam">
            <video ref={cam.videoRef} muted playsInline />
            <div className="roi" style={{ left: `${cam.roi.x * 100}%`, top: `${cam.roi.y * 100}%`, width: `${cam.roi.w * 100}%`, height: `${cam.roi.h * 100}%` }} />
            {cam.error && <p className="error">{cam.error}</p>}
            <p className="hint">Put your forehead in the box, keep still, steady light.</p>
          </div>
        )}
        {session.vitals_source === 'presage' && (
          <p className="hint">Waiting for the Presage bridge to POST samples to /api/sessions/{sid}/vitals.</p>
        )}
        <div className="vitals">
          <div><b>{live?.hr ?? '–'}</b><span>bpm</span></div>
          <div><b>{live?.br ?? '–'}</b><span>breaths/min</span></div>
          <div><b>{live ? (live.hr_z >= 0 ? '+' : '') + live.hr_z : '–'}</b><span>σ vs your baseline</span></div>
        </div>
        <Sparkline values={hrs} />
        <div className="meter">
          <div className="meter-label">Shark's bluff read if you bet now</div>
          <div className="bar"><div style={{ width: `${Math.round(read * 100)}%` }} className={read > 0.6 ? 'hot' : ''} /></div>
          <div className="meter-value">{Math.round(read * 100)}%{live && !live.shark_ready ? ' (still learning)' : ''}</div>
        </div>
        <p className="coach">{live?.line ?? 'Sit still for 45 seconds so we can learn your resting baseline.'}</p>
        <label className="toggle"><input type="checkbox" checked={voice} onChange={(e) => setVoice(e.target.checked)} /> Voice coach</label>
      </aside>
    </div>
  )
}
