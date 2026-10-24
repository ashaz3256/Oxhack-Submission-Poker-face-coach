async function call(method, path, body) {
  const res = await fetch(`/api${path}`, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText)
  return res.status === 204 ? null : res.json()
}

export const api = {
  createSession: (opts) => call('POST', '/sessions', opts),
  newHand: (sid) => call('POST', `/sessions/${sid}/hands`),
  act: (sid, action) => call('POST', `/sessions/${sid}/action`, { action }),
  simTick: (sid) => call('POST', `/sessions/${sid}/sim-tick`),
  sendVitals: (sid, samples) => call('POST', `/sessions/${sid}/vitals`, { samples }),
  live: (sid) => call('GET', `/sessions/${sid}/live`),
  report: (sid) => call('GET', `/sessions/${sid}/report`),
  review: (sid) => call('POST', `/sessions/${sid}/review`),
  demo: (hands = 60) => call('POST', `/demo?hands=${hands}`),
}

let speaking = false
export async function speak(text) {
  if (speaking) return
  speaking = true
  try {
    const res = await fetch('/api/speak', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    })
    if (res.status === 200) {
      const audio = new Audio(URL.createObjectURL(await res.blob()))
      await audio.play()
      await new Promise((r) => (audio.onended = r))
    } else if ('speechSynthesis' in window) {
      await new Promise((r) => {
        const u = new SpeechSynthesisUtterance(text)
        u.onend = r
        window.speechSynthesis.speak(u)
      })
    }
  } catch {
    // voice is a nice-to-have; never break the table over it
  } finally {
    speaking = false
  }
}
