import { useCallback, useEffect, useRef, useState } from 'react'
import { mafia, speak } from './api.js'
import { useWebcamPulse } from './useWebcamPulse.js'
import { Sparkline } from './Table.jsx'

const AVATARS = { You: '🫵', Ava: '👩‍🦰', Ben: '🧔', Cleo: '👩‍🦱', Dev: '👨‍🦲', Eli: '🧑‍🦳' }

function Players({ players, pick, onPick }) {
  return (
    <div className="players">
      {players.map((p) => {
        const canPick = pick && p.alive && p.name !== 'You'
        return (
          <button key={p.name} disabled={!canPick} onClick={() => canPick && onPick(p.name)}
            className={`player ${p.alive ? '' : 'dead'} ${p.role === 'mafia' ? 'mafia' : ''} ${canPick ? 'pickable' : ''}`}>
            <span className="avatar">{AVATARS[p.name] || '🙂'}</span>
            <span className="pname">{p.name}</span>
            <span className="prole">{p.alive ? (p.role ?? '?') : `✝ ${p.role ?? ''}`}</span>
          </button>
        )
      })}
    </div>
  )
}

export default function MafiaTable({ session, onReport }) {
  const sid = session.id
  const [state, setState] = useState(null)
  const [live, setLive] = useState(null)
  const [hrs, setHrs] = useState([])
  const [voice, setVoice] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const lastSpoken = useRef('')
  const pending = useRef([])
  const chatEnd = useRef(null)

  const onSample = useCallback((s) => {
    pending.current.push(s)
    setHrs((h) => [...h.slice(-90), s.hr])
  }, [])
  const cam = useWebcamPulse(session.vitals_source === 'webcam', onSample)

  useEffect(() => {
    const id = setInterval(async () => {
      try {
        if (session.vitals_source === 'sim') {
          const s = await mafia.simTick(sid)
          setHrs((h) => [...h.slice(-90), s.hr])
        } else if (pending.current.length) {
          const batch = pending.current
          pending.current = []
          await mafia.sendVitals(sid, batch)
        }
        setLive(await mafia.live(sid))
      } catch {
        // keep ticking
      }
    }, 1000)
    return () => clearInterval(id)
  }, [sid, session.vitals_source])

  useEffect(() => {
    mafia.newGame(sid).then(setState).catch((e) => setError(e.message))
  }, [sid])

  const game = state?.game
  useEffect(() => {
    chatEnd.current?.scrollIntoView({ block: 'nearest' })
    // Read the Inspector's questions aloud.
    const last = [...(game?.chat ?? [])].reverse().find((c) => c.speaker === 'Inspector')
    const key = `${game?.number}-${game?.chat.length}-${last?.text}`
    if (voice && last && game?.phase === 'question' && lastSpoken.current !== key) {
      lastSpoken.current = key
      speak(last.text)
    }
  }, [game, voice])

  const step = async (fn) => {
    setBusy(true)
    setError(null)
    try {
      const res = await fn()
      setState(res)
      if (res.live) setLive(res.live)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  const read = live?.p_lie ?? 0
  const phase = game?.phase

  return (
    <div className="table-layout">
      <section className="panel mafia-board">
        <div className="mafia-top">
          <div>
            <div className="who">Game {game?.number ?? '–'} · Day {game?.day ?? '–'}</div>
            <div className={`role-badge ${game?.you_mafia ? 'mafia' : ''}`}>
              {game ? (game.you_mafia ? 'You are the MAFIA' : 'You are a VILLAGER') : ''}
            </div>
          </div>
          <div className="who">Record: {state?.record.wins ?? 0} wins / {state?.record.games ?? 0} games</div>
        </div>

        <Players players={game?.players ?? []} pick={phase === 'vote' || phase === 'night'}
          onPick={(name) => step(() => (phase === 'vote' ? mafia.vote(sid, name) : mafia.kill(sid, name)))} />

        <div className="chat">
          {game?.chat.map((c, i) => (
            <div key={i} className={`line ${c.speaker === 'You' ? 'me' : ''} ${c.speaker === 'Inspector' ? 'inspector' : ''} ${c.speaker === 'Narrator' ? 'narrator' : ''}`}>
              <b>{c.speaker}</b> {c.text}
            </div>
          ))}
          <div ref={chatEnd} />
        </div>

        <div className="prompt">
          {phase === 'question' && (
            <>
              <div className="q">🕵️ {game.question.text}</div>
              <div className="answers">
                {game.question.options.map((o) => (
                  <button key={o.id} disabled={busy} onClick={() => step(() => mafia.answer(sid, o.id))}>{o.text}</button>
                ))}
              </div>
            </>
          )}
          {phase === 'vote' && <div className="q">Vote: click the player you want eliminated.</div>}
          {phase === 'night' && <div className="q">🌙 Night. As the mafia, click who to eliminate.</div>}
          {phase === 'over' && (
            <div className="answers">
              <button className="primary" onClick={() => step(() => mafia.newGame(sid))}>Next game</button>
              <button onClick={onReport}>Leak report</button>
            </div>
          )}
        </div>
        {error && <p className="error">{error}</p>}
      </section>

      <aside className="panel side">
        <h3>What the Inspector sees</h3>
        {session.vitals_source === 'webcam' && (
          <div className="cam">
            <video ref={cam.videoRef} muted playsInline />
            <div className="roi" style={{ left: `${cam.roi.x * 100}%`, top: `${cam.roi.y * 100}%`, width: `${cam.roi.w * 100}%`, height: `${cam.roi.h * 100}%` }} />
            {cam.error && <p className="error">{cam.error}</p>}
            <p className="hint">Put your forehead in the box, keep still, steady light.</p>
          </div>
        )}
        {session.vitals_source === 'presage' && (
          <p className="hint">Waiting for the Presage bridge to POST samples to /api/mafia/sessions/{sid}/vitals.</p>
        )}
        <div className="vitals">
          <div><b>{live?.hr ?? '–'}</b><span>bpm</span></div>
          <div><b>{live?.br ?? '–'}</b><span>breaths/min</span></div>
          <div><b>{live ? (live.hr_z >= 0 ? '+' : '') + live.hr_z : '–'}</b><span>σ vs your baseline</span></div>
        </div>
        <Sparkline values={hrs} />
        <div className="meter">
          <div className="meter-label">
            {session.inspector === 'reader' ? "Inspector's lie read right now" : 'Lie read (hidden from this Inspector)'}
          </div>
          <div className="bar"><div style={{ width: `${Math.round(read * 100)}%` }} className={read > 0.6 ? 'hot' : ''} /></div>
          <div className="meter-value">{Math.round(read * 100)}%{live && !live.ready ? ' (still learning)' : ''}</div>
        </div>
        <p className="coach">{live?.line ?? 'Sit still for 45 seconds so we can learn your resting baseline.'}</p>
        <label className="toggle"><input type="checkbox" checked={voice} onChange={(e) => setVoice(e.target.checked)} /> Inspector's voice</label>
      </aside>
    </div>
  )
}
