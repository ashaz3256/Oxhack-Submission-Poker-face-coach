import { useState } from 'react'
import { api } from './api.js'
import Table from './Table.jsx'
import Report from './Report.jsx'

export default function App() {
  const [session, setSession] = useState(null)
  const [view, setView] = useState('setup')
  const [opts, setOpts] = useState({ opponent: 'shark', vitals_source: 'sim', leakiness: 1, player: '' })
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  const start = async () => {
    setBusy(true)
    setError(null)
    try {
      const res = await api.createSession({ ...opts, player: opts.player || 'player' })
      setSession({ id: res.session_id, ...opts })
      setView('table')
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  const demo = async () => {
    setBusy(true)
    setError(null)
    try {
      const res = await api.demo(60)
      setSession({ id: res.session_id, opponent: 'shark', vitals_source: 'sim' })
      setView('report')
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="app">
      <header>
        <h1>Tilt</h1>
        <p className="tag">Your body is leaking information. Find out how much it costs you.</p>
        {session && (
          <nav>
            <button className={view === 'table' ? 'on' : ''} onClick={() => setView('table')}>Table</button>
            <button className={view === 'report' ? 'on' : ''} onClick={() => setView('report')}>Leak report</button>
            <button onClick={() => { setSession(null); setView('setup') }}>New session</button>
          </nav>
        )}
      </header>

      {view === 'setup' && (
        <section className="setup panel">
          <h2>Sit down</h2>
          <label>Name
            <input value={opts.player} placeholder="player" onChange={(e) => setOpts({ ...opts, player: e.target.value })} />
          </label>
          <label>Opponent
            <select value={opts.opponent} onChange={(e) => setOpts({ ...opts, opponent: e.target.value })}>
              <option value="shark">Shark: reads your vitals</option>
              <option value="honest">Honest: only knows your bluff rate</option>
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
          <p className="hint">Both bots bet the same way. The only difference: when you bet, the Shark asks your heart how likely you are to be bluffing. We score every bet against both, so the gap is what your body costs you.</p>
          <div className="row">
            <button className="primary" disabled={busy} onClick={start}>Deal me in</button>
            <button disabled={busy} onClick={demo}>{busy ? 'Simulating 60 hands…' : 'Demo report (simulated)'}</button>
          </div>
          {error && <p className="error">{error}</p>}
        </section>
      )}

      {view === 'table' && session && <Table session={session} onReport={() => setView('report')} />}
      {view === 'report' && session && <Report sessionId={session.id} />}
    </div>
  )
}
