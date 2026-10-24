// After a game: every answer, vote and kill, marked as what you got right and where it went wrong,
// with your biggest tells and what to do about them.
import { useEffect, useState } from 'react'
import { api } from './api.js'

const MARK = { good: '✓', bad: '✗', neutral: '•' }

function Moment({ m }) {
  return (
    <div className={`moment ${m.verdict}`}>
      <div className="moment-head">
        <span className="moment-mark">{MARK[m.verdict]}</span>
        <b>{m.title}</b>
        {m.p_lie != null && <span className="moment-p">{Math.round(m.p_lie * 100)}% lie</span>}
      </div>
      {m.question && <div className="moment-q">Q: {m.question}</div>}
      {m.type === 'answer' && m.said && <div className="moment-said">“{m.said}”</div>}
      {m.p_lie != null && (
        <div className="bar"><div className={m.p_lie >= 0.5 ? 'hot' : ''} style={{ width: `${Math.round(m.p_lie * 100)}%` }} /></div>
      )}
      {m.detail && <div className="moment-detail">{m.detail}</div>}
    </div>
  )
}

export default function Breakdown({ sessionId, refreshKey, review }) {
  const [game, setGame] = useState(null)
  const [b, setB] = useState(null)
  useEffect(() => {
    api.breakdown(sessionId, game).then(setB).catch(() => {})
  }, [sessionId, refreshKey, game])
  if (!b) return <p className="hint">Loading your breakdown…</p>
  if (b.game == null) return <p className="hint">Finish a game to see your breakdown.</p>
  const days = [...new Set(b.moments.map((m) => m.day))]
  const mafia = b.role === 'mafia'
  return (
    <div className="breakdown">
      <div className={`bd-banner ${b.you_won ? 'won' : 'lost'}`}>
        <div className="bd-result">{b.over ? (b.you_won ? 'You won' : 'You lost') : 'In progress'}</div>
        <div className="bd-sub">
          Game {b.game} · you were <span className={`role ${b.role}`}>{b.role}</span>
          {!mafia && b.mafia && <> · the mafia was <b>{b.mafia}</b></>}
          {' '}· {b.survived ? 'you survived' : 'you died'}
        </div>
        <div className="bd-score">{b.score_line}</div>
      </div>
      {b.games.length > 1 && (
        <div className="bd-games">
          {b.games.map((n) => (
            <button key={n} className={n === b.game ? 'on' : ''} onClick={() => setGame(n)}>Game {n}</button>
          ))}
        </div>
      )}
      {review && <div className="bd-coach"><b>Coach:</b> {review}</div>}
      <div className="bd-cols">
        <div className="bd-col good">
          <h3>What you did right</h3>
          {b.right.length ? <ul>{b.right.map((t, i) => <li key={i}>{t}</li>)}</ul> : <p className="hint">Nothing went your way this time.</p>}
        </div>
        <div className="bd-col bad">
          <h3>Where you went wrong</h3>
          {b.wrong.length ? <ul>{b.wrong.map((t, i) => <li key={i}>{t}</li>)}</ul> : <p className="hint">Nothing. Clean game.</p>}
        </div>
      </div>
      {b.tells.length > 0 && (
        <>
          <h3>{mafia ? 'What gave your lies away' : 'What made you look guilty'}</h3>
          <div className="bd-tells">
            {b.tells.map((t) => (
              <div key={t.key} className="bd-tell">
                <div><b>{t.label}</b> <span className="hint">{t.how}</span></div>
                <div className="bd-tip">{t.tip}</div>
              </div>
            ))}
          </div>
        </>
      )}
      <h3>Play by play</h3>
      {days.map((d) => (
        <div key={d} className="bd-day">
          <div className="bd-day-title">Day {d}</div>
          {b.moments.filter((m) => m.day === d).map((m, i) => <Moment key={i} m={m} />)}
        </div>
      ))}
    </div>
  )
}
