import { useEffect, useState } from 'react'
import { api, speak } from './api.js'

const FEATURE_NAMES = {
  hr_z: 'Heart rate vs baseline',
  hr_slope: 'Heart rate rising',
  br_z: 'Breathing vs baseline',
  decision_s: 'Time to act',
  last_result_bb: 'Last hand result',
  drawdown_bb: 'Drawdown',
}

function CostCurve({ curve }) {
  const W = 1000
  const H = 240
  const pad = 36
  if (curve.length < 2) return <p className="hint">Not enough bets yet.</p>
  const ys = curve.map((c) => c.cost_bb)
  const lo = Math.min(0, ...ys)
  const hi = Math.max(1, ...ys)
  const x = (i) => pad + (i / (curve.length - 1)) * (W - pad - 12)
  const y = (v) => H - pad - ((v - lo) / (hi - lo)) * (H - pad - 12)
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label="Cumulative big blinds lost to tells, bet by bet">
      <line x1={pad} x2={W - 12} y1={y(0)} y2={y(0)} className="zero" />
      <polyline points={curve.map((c, i) => `${x(i)},${y(c.cost_bb)}`).join(' ')} className="line" />
      {curve.map((c, i) => c.bluff && <circle key={i} cx={x(i)} cy={y(c.cost_bb)} r="3" className="dot" />)}
      <text x={pad} y={14} className="axis">bb lost to tells</text>
      <text x={W - 12} y={H - 10} textAnchor="end" className="axis">your bets, in order ({curve.length})</text>
      <text x={pad - 6} y={y(hi) + 4} textAnchor="end" className="axis">{hi.toFixed(0)}</text>
      <text x={pad - 6} y={y(0) + 4} textAnchor="end" className="axis">0</text>
    </svg>
  )
}

function Coefficients({ coef }) {
  const entries = Object.entries(coef || {}).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
  if (!entries.length) return <p className="hint">The model needs a few bluffs and a few value bets first.</p>
  const max = Math.max(...entries.map(([, v]) => Math.abs(v)))
  return (
    <div className="coefs">
      {entries.map(([k, v]) => (
        <div key={k} className="coef">
          <span>{FEATURE_NAMES[k] || k}</span>
          <div className="bar"><div style={{ width: `${(Math.abs(v) / max) * 100}%` }} className={v > 0 ? 'hot' : 'cool'} /></div>
          <span className="num">{v > 0 ? '+' : ''}{v}</span>
        </div>
      ))}
    </div>
  )
}

export default function Report({ sessionId }) {
  const [r, setR] = useState(null)
  const [review, setReview] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    api.report(sessionId).then(setR)
  }, [sessionId])

  const getReview = async () => {
    setLoading(true)
    try {
      const res = await api.review(sessionId)
      setReview(res)
      speak(res.text)
    } finally {
      setLoading(false)
    }
  }

  if (!r) return <p>Loading…</p>
  const t = r.tells
  return (
    <div className="report">
      <section className="panel">
        <h2>{r.headline}</h2>
        <div className="tiles">
          <div className="tile"><b>{t.cost_bb}{t.cost_se_bb != null && <small> ± {t.cost_se_bb}</small>}</b><span>bb your tells cost you</span></div>
          <div className="tile"><b>{t.shark_auc ?? '–'}</b><span>Shark's bluff-read AUC (0.5 = coin flip)</span></div>
          <div className="tile"><b>{t.bluffs}/{t.bets}</b><span>of your bets were bluffs</span></div>
          <div className="tile"><b>{r.bb_per_100}</b><span>bb/100 vs {r.opponent}, {r.hands} hands</span></div>
        </div>
        <p className="hint">Every bet is scored twice: once against the Shark, once against an Honest bot that bets identically but only knows your overall bluff rate. The gap is the cost of your tells. The Shark's read is taken before it learns each answer, so the AUC is out-of-sample.</p>
      </section>

      <section className="panel">
        <h3>The Shark learning you</h3>
        <CostCurve curve={t.curve} />
        <p className="hint">Dots mark bluffs. A line that climbs means the Shark is calling your bluffs and folding to your value bets.</p>
      </section>

      <div className="two">
        <section className="panel">
          <h3>What gives you away</h3>
          <Coefficients coef={t.coefficients} />
          <table>
            <thead><tr><th /><th>Bluffing</th><th>Value</th></tr></thead>
            <tbody>
              <tr><td>Heart rate (bpm)</td><td>{t.means.hr.bluff ?? '–'}</td><td>{t.means.hr.value ?? '–'}</td></tr>
              <tr><td>vs baseline (σ)</td><td>{t.means.hr_z.bluff ?? '–'}</td><td>{t.means.hr_z.value ?? '–'}</td></tr>
              <tr><td>Breathing (σ)</td><td>{t.means.br_z.bluff ?? '–'}</td><td>{t.means.br_z.value ?? '–'}</td></tr>
              <tr><td>Time to act (s)</td><td>{t.means.decision_s.bluff ?? '–'}</td><td>{t.means.decision_s.value ?? '–'}</td></tr>
            </tbody>
          </table>
        </section>
        <section className="panel">
          <h3>Tilt: bad calls and folds</h3>
          <div className="tiles small">
            <div className="tile"><b>{r.tilt.bad}/{r.tilt.decisions}</b><span>bad decisions facing a bet</span></div>
            <div className="tile"><b>{r.tilt.bad_rate_stressed ?? '–'}</b><span>bad rate when HR &gt; +1σ</span></div>
            <div className="tile"><b>{r.tilt.bad_rate_calm ?? '–'}</b><span>bad rate when calm</span></div>
            <div className="tile"><b>{r.tilt.model_auc ?? '–'}</b><span>tilt model AUC</span></div>
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
