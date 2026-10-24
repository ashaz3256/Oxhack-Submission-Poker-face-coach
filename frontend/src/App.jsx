import { useState } from 'react'
import { api, mafia } from './api.js'
import Table from './Table.jsx'
import Report from './Report.jsx'
import MafiaTable from './MafiaTable.jsx'
import MafiaReport from './MafiaReport.jsx'

const COPY = {
  mafia: {
    opponentLabel: 'Inspector',
    opponents: [
      ['reader', 'Reader: reads your vitals'],
      ['blind', 'Blind: only knows how often you lie'],
    ],
    hint: "You play Mafia against five AI players. The Inspector questions you every day, and when you're the mafia every denial is a lie. The Reader Inspector checks your pulse while you answer; the Blind one doesn't. We replay every vote both ways, so the gap is what your body costs you.",
    demo: 'Simulating 15 games…',
  },
  poker: {
    opponentLabel: 'Opponent',
    opponents: [
      ['shark', 'Shark: reads your vitals'],
      ['honest', 'Honest: only knows your bluff rate'],
    ],
    hint: 'Both bots bet the same way. The only difference: when you bet, the Shark asks your heart how likely you are to be bluffing. We score every bet against both, so the gap is what your body costs you.',
    demo: 'Simulating 60 hands…',
  },
}

export default function App() {
  const [session, setSession] = useState(null)
  const [view, setView] = useState('setup')
  const [game, setGame] = useState('mafia')
  const [opts, setOpts] = useState({ opponent: 'reader', vitals_source: 'sim', leakiness: 1, player: '' })
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const copy = COPY[game]

  const pickGame = (g) => {
    setGame(g)
    setOpts({ ...opts, opponent: COPY[g].opponents[0][0] })
  }

  const run = async (fn) => {
    setBusy(true)
    setError(null)
    try {
      await fn()
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  const start = () => run(async () => {
    const base = { vitals_source: opts.vitals_source, leakiness: opts.leakiness, player: opts.player || 'player' }
    const res = game === 'mafia'
      ? await mafia.create({ ...base, inspector: opts.opponent })
      : await api.createSession({ ...base, opponent: opts.opponent })
    setSession({ id: res.session_id, game, ...opts, inspector: opts.opponent })
    setView('table')
  })

  const demo = () => run(async () => {
    const res = game === 'mafia' ? await mafia.demo(15) : await api.demo(60)
    setSession({ id: res.session_id, game, opponent: copy.opponents[0][0], vitals_source: 'sim' })
    setView('report')
  })

  return (
    <div className="app">
      <header>
        <h1>Tilt</h1>
        <p className="tag">Your body is leaking information. Find out how much it costs you.</p>
        {session && (
          <nav>
            <button className={view === 'table' ? 'on' : ''} onClick={() => setView('table')}>Play</button>
            <button className={view === 'report' ? 'on' : ''} onClick={() => setView('report')}>Leak report</button>
            <button onClick={() => { setSession(null); setView('setup') }}>New session</button>
          </nav>
        )}
      </header>

      {view === 'setup' && (
        <section className="setup panel">
          <h2>Sit down</h2>
          <div className="games">
            <button className={game === 'mafia' ? 'on' : ''} onClick={() => pickGame('mafia')}>Mafia</button>
            <button className={game === 'poker' ? 'on' : ''} onClick={() => pickGame('poker')}>Poker</button>
          </div>
          <label>Name
            <input value={opts.player} placeholder="player" onChange={(e) => setOpts({ ...opts, player: e.target.value })} />
          </label>
          <label>{copy.opponentLabel}
            <select value={opts.opponent} onChange={(e) => setOpts({ ...opts, opponent: e.target.value })}>
              {copy.opponents.map(([v, label]) => <option key={v} value={v}>{label}</option>)}
            </select>
          </label>
          <label>Vitals
            <select value={opts.vitals_source} onChange={(e) => setOpts({ ...opts, vitals_source: e.target.value })}>
              <option value="sim">Simulated</option>
              <option value="webcam">Webcam pulse (rPPG)</option>
              <option value="presage">Presage bridge (POSTs to /vitals)</option>
            </select>
          </label>
          {opts.vitals_source === 'sim' && (
            <label>Leakiness {opts.leakiness}
              <input type="range" min="0" max="2" step="0.25" value={opts.leakiness}
                onChange={(e) => setOpts({ ...opts, leakiness: Number(e.target.value) })} />
            </label>
          )}
          <p className="hint">{copy.hint}</p>
          <div className="row">
            <button className="primary" disabled={busy} onClick={start}>Deal me in</button>
            <button disabled={busy} onClick={demo}>{busy ? copy.demo : 'Demo report (simulated)'}</button>
          </div>
          {error && <p className="error">{error}</p>}
        </section>
      )}

      {view === 'table' && session?.game === 'mafia' && <MafiaTable session={session} onReport={() => setView('report')} />}
      {view === 'table' && session?.game === 'poker' && <Table session={session} onReport={() => setView('report')} />}
      {view === 'report' && session?.game === 'mafia' && <MafiaReport sessionId={session.id} />}
      {view === 'report' && session?.game === 'poker' && <Report sessionId={session.id} />}
    </div>
  )
}
