import { useCallback, useState } from 'react'
import { api } from './api.js'
import Call from './Call.jsx'
import Report from './Report.jsx'
import Splash from './Splash.jsx'
import { unlockAudio } from './tts.js'

const params = new URLSearchParams(window.location.search)
const dev = params.has('dev')
const skipSplash = params.has('fast')

export default function App() {
  const [screen, setScreen] = useState('lobby')
  const [demoId, setDemoId] = useState(null)
  const [busy, setBusy] = useState(false)
  const [splash, setSplash] = useState(!skipSplash)
  const endSplash = useCallback(() => setSplash(false), [])

  const join = () => {
    unlockAudio() // browsers only allow sound and mic after one click
    setScreen('call')
  }

  const demo = async () => {
    setBusy(true)
    try {
      setDemoId((await api.demo()).session_id)
      setScreen('demo')
    } finally {
      setBusy(false)
    }
  }

  if (screen === 'call') return <Call dev={dev} />
  if (screen === 'demo') {
    return (
      <div className="app">
        <button onClick={() => setScreen('lobby')}>← Back</button>
        <div className="panel"><Report sessionId={demoId} /></div>
      </div>
    )
  }
  return (
    <div className={`lobby-scene ${splash ? '' : 'revealed'}`}>
      {splash && <Splash onDone={endSplash} />}
      <div className="intro-stars" />
      <div className="lobby-moon" />
      <div className="intro-fog" />
      <div className="lobby-cards" aria-hidden="true">
        {[1, 2, 3, 4, 5, 6].map((n, i) => <span key={n} style={{ '--i': i }}>{n}</span>)}
      </div>
    <div className="lobby">
      <h1 className="creepy">Tilt</h1>
      <p className="tag">Mafia on a call with five AIs. Every word you say is checked for lies.</p>
      <ol className="how">
        <li>Everyone draws a card. Yours says <b>mafia</b> or <b>villager</b>.</li>
        <li>The AIs talk, accuse and question you. You just answer out loud.</li>
        <li>We scan your eyes, brows and mouth, your pitch and speed, and your words. A model learns when you lie.</li>
        <li>The AIs hear your tells. Lie badly and they vote you out.</li>
      </ol>
      <button className="primary big join" onClick={join}>🩸 Join the call</button>
      <p className="hint">Needs a webcam and microphone. Hands-free from here on. Chrome or Edge work best.</p>
      <button className="link" disabled={busy} onClick={demo}>{busy ? 'Simulating 15 games…' : 'See a demo report (simulated player)'}</button>
    </div>
    </div>
  )
}
