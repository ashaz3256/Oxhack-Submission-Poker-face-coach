import { useState } from 'react'
import { api } from './api.js'
import Call from './Call.jsx'
import Report from './Report.jsx'
import { sttSupported } from './sensors/stt.js'
import { unlockAudio } from './tts.js'

const dev = new URLSearchParams(window.location.search).has('dev')

export default function App() {
  const [screen, setScreen] = useState('lobby')
  const [demoId, setDemoId] = useState(null)
  const [busy, setBusy] = useState(false)

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
    <div className="lobby">
      <h1>Tilt</h1>
      <p className="tag">Mafia on a call with five AIs. Every word you say is checked for lies.</p>
      <ol className="how">
        <li>Everyone draws a card. Yours says <b>mafia</b> or <b>villager</b>.</li>
        <li>The AIs talk, accuse and question you. You just answer out loud.</li>
        <li>We scan your eyes, brows and mouth, your pitch and speed, and your words. A model learns when you lie.</li>
        <li>The AIs hear your tells. Lie badly and they vote you out.</li>
      </ol>
      <button className="primary big" onClick={join}>Join the call</button>
      <p className="hint">Needs a webcam and microphone. {sttSupported ? 'Hands-free from here on.' : 'Use Chrome or Edge for speech recognition.'}</p>
      <button className="link" disabled={busy} onClick={demo}>{busy ? 'Simulating 15 games…' : 'See a demo report (simulated player)'}</button>
    </div>
  )
}
