import { useEffect, useState } from 'react'
import { api } from './api.js'

function Timeline({ rows }) {
  if (!rows.length) return <p className="hint">No answers yet.</p>
  return (
    <div className="timeline">
      {rows.map((r) => (
        <div key={r.i} className="tl-row" title={r.cues.join(', ')}>
          <span className={`tl-tag ${r.lie ? 'lie' : 'truth'}`}>{r.lie ? 'LIE' : 'TRUE'}</span>
          <div className="bar"><div style={{ width: `${Math.round(r.p_lie * 100)}%` }} className={r.p_lie > 0.6 ? 'hot' : ''} /></div>
          <span className="num">{Math.round(r.p_lie * 100)}%</span>
          <span className="tl-text">“{r.text}”</span>
        </div>
      ))}
    </div>
  )
}

function Coefficients({ coef }) {
  const entries = Object.entries(coef || {}).sort((a, b) => b[1] - a[1]).slice(0, 8)
  if (!entries.length) return <p className="hint">The model needs a few lies and a few truths first.</p>
  const max = Math.max(...entries.map(([, v]) => Math.abs(v)))
  return (
    <div className="coefs">
      {entries.map(([k, v]) => (
        <div key={k} className="coef">
          <span>{k}</span>
          <div className="bar"><div style={{ width: `${(Math.abs(v) / max) * 100}%` }} className={v > 0 ? 'hot' : 'cool'} /></div>
          <span className="num">{v > 0 ? '+' : ''}{v}</span>
        </div>
      ))}
    </div>
  )
}

const fmt = (v) => (v == null ? '–' : Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(2))

export default function Report({ sessionId, refreshKey }) {
  const [r, setR] = useState(null)
  useEffect(() => {
    api.report(sessionId).then(setR).catch(() => {})
  }, [sessionId, refreshKey])
  if (!r) return <p className="hint">Loading report…</p>
  return (
    <div className="report">
      <h2>{r.headline}</h2>
      <div className="tiles">
        <div className="tile"><b>{r.practice_accuracy != null ? `${Math.round(r.practice_accuracy * 100)}%` : '–'}</b><span>accuracy on your warm-up (cross-validated)</span></div>
        <div className="tile"><b>{r.game_accuracy != null ? `${Math.round(r.game_accuracy * 100)}%` : '–'}</b><span>accuracy in the game (read before it learned)</span></div>
        <div className="tile"><b>{r.lie_auc ?? '–'}</b><span>lie-detection AUC (0.5 = coin flip)</span></div>
        <div className="tile"><b>{r.caught_by_reader} vs {r.caught_by_blind}</b><span>days caught as mafia: read vs unread</span></div>
        <div className="tile"><b>{r.lies}/{r.answers}</b><span>answers that were lies</span></div>
        <div className="tile"><b>{r.mafia_wins}/{r.games_as_mafia}</b><span>games won as mafia</span></div>
      </div>
      <h3>Every answer, as the detector read it</h3>
      <Timeline rows={r.timeline} />
      <h3>What gives you away</h3>
      <Coefficients coef={r.coefficients} />
      <table>
        <thead><tr><th>Cue</th><th>Lying</th><th>Truthful</th></tr></thead>
        <tbody>
          {r.cues.map((c) => (
            <tr key={c.key}><td>{c.label}</td><td>{fmt(c.lie)}</td><td>{fmt(c.truth)}</td></tr>
          ))}
        </tbody>
      </table>
      <p className="hint">Each read is made before the model learns whether that answer was a lie, so the AUC is out-of-sample. “Caught” compares the real vote with the same vote replayed without your tells.</p>
    </div>
  )
}
