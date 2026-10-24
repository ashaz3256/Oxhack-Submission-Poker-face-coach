import { useEffect, useState } from 'react'
import { mafia, speak } from './api.js'
import { Coefficients } from './Report.jsx'

function ReadStrip({ curve }) {
  if (curve.length < 2) return <p className="hint">Not enough answers yet.</p>
  const W = 1000
  const H = 220
  const pad = 40
  const x = (i) => pad + (i / (curve.length - 1)) * (W - pad - 16)
  const y = (p) => H - pad - p * (H - pad - 16)
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label="The Inspector's lie read for each answer">
      <line x1={pad} x2={W - 16} y1={y(0.5)} y2={y(0.5)} className="zero" strokeDasharray="4 4" />
      {curve.map((c, i) => (
        <circle key={i} cx={x(i)} cy={y(c.p_lie)} r="6" className={c.lie ? 'dot-lie' : 'dot-truth'} />
      ))}
      <text x={pad - 8} y={y(1) + 4} textAnchor="end" className="axis">100%</text>
      <text x={pad - 8} y={y(0.5) + 4} textAnchor="end" className="axis">50%</text>
      <text x={pad - 8} y={y(0) + 4} textAnchor="end" className="axis">0%</text>
      <text x={W - 16} y={H - 8} textAnchor="end" className="axis">your answers, in order ({curve.length})</text>
    </svg>
  )
}

export default function MafiaReport({ sessionId }) {
  const [r, setR] = useState(null)
  const [review, setReview] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    mafia.report(sessionId).then(setR)
  }, [sessionId])

  const getReview = async () => {
    setLoading(true)
    try {
      const res = await mafia.review(sessionId)
      setReview(res)
      speak(res.text)
    } finally {
      setLoading(false)
    }
  }

  if (!r) return <p>Loading…</p>
  return (
    <div className="report">
      <section className="panel">
        <h2>{r.headline}</h2>
        <div className="tiles">
          <div className="tile"><b>{r.caught_by_reader} vs {r.caught_by_blind}</b><span>days voted out as mafia: Inspector reading you vs blind</span></div>
          <div className="tile"><b>{r.votes_on_you_reader ?? '–'} vs {r.votes_on_you_blind ?? '–'}</b><span>votes on you per day as mafia</span></div>
          <div className="tile"><b>{r.lie_auc ?? '–'}</b><span>lie-detection AUC (0.5 = coin flip)</span></div>
          <div className="tile"><b>{r.lies}/{r.answers}</b><span>of your claims were lies</span></div>
        </div>
        <p className="hint">Every day is voted twice with the same dice: once by a town whose Inspector reads your vitals, once by a town whose Inspector only knows how often you lie. The gap is what your body cost you. Each lie read is made before the Inspector learns the answer, so the AUC is out-of-sample.</p>
      </section>

      <section className="panel">
        <h3>The Inspector learning you</h3>
        <ReadStrip curve={r.curve} />
        <p className="hint"><span className="key lie" /> lie <span className="key truth" /> truth. Lies drifting above the dashed line and truths below it means your body is talking.</p>
      </section>

      <div className="two">
        <section className="panel">
          <h3>What gives you away</h3>
          <Coefficients coef={r.coefficients} />
          <table>
            <thead><tr><th /><th>Lying</th><th>Truthful</th></tr></thead>
            <tbody>
              <tr><td>Heart rate (bpm)</td><td>{r.means.hr.lie ?? '–'}</td><td>{r.means.hr.truth ?? '–'}</td></tr>
              <tr><td>vs baseline (σ)</td><td>{r.means.hr_z.lie ?? '–'}</td><td>{r.means.hr_z.truth ?? '–'}</td></tr>
              <tr><td>Breathing (σ)</td><td>{r.means.br_z.lie ?? '–'}</td><td>{r.means.br_z.truth ?? '–'}</td></tr>
              <tr><td>Time to answer (s)</td><td>{r.means.decision_s.lie ?? '–'}</td><td>{r.means.decision_s.truth ?? '–'}</td></tr>
            </tbody>
          </table>
        </section>
        <section className="panel">
          <h3>Your games</h3>
          <div className="tiles small">
            <div className="tile"><b>{r.games}</b><span>games played</span></div>
            <div className="tile"><b>{r.games_as_mafia}</b><span>as the mafia</span></div>
            <div className="tile"><b>{r.mafia_wins}</b><span>won as the mafia</span></div>
            <div className="tile"><b>{r.extra_votes_from_tells}</b><span>extra votes your tells drew</span></div>
          </div>
        </section>
      </div>

      <section className="panel">
        <h3>Coach's review</h3>
        <button className="primary" disabled={loading} onClick={getReview}>{loading ? 'Thinking…' : 'Review my session'}</button>
        {review && <p className="coach">{review.text} <small>({review.source})</small></p>}
      </section>
    </div>
  )
}
