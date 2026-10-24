async function call(method, path, body) {
  const res = await fetch(`/api${path}`, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText)
  return res.status === 204 ? null : res.json()
}

async function transcribe(blob) {
  const res = await fetch('/api/transcribe', { method: 'POST', headers: { 'Content-Type': 'audio/wav' }, body: blob })
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText)
  return (await res.json()).text
}

export const api = {
  transcribe,
  create: () => call('POST', '/sessions', {}),
  calibrate: (sid, transcript, raw, isLie, question, face_frames = null) => call('POST', `/sessions/${sid}/calibrate`, { transcript, raw, is_lie: isLie, question, face_frames }),
  newGame: (sid) => call('POST', `/sessions/${sid}/games`),
  speak: (sid, transcript, raw, face_frames = null) => call('POST', `/sessions/${sid}/speak`, { transcript, raw, face_frames }),
  faceModel: () => call('GET', '/face-model'),
  report: (sid) => call('GET', `/sessions/${sid}/report`),
  breakdown: (sid, game) => call('GET', `/sessions/${sid}/breakdown${game ? `?game=${game}` : ''}`),
  review: (sid) => call('POST', `/sessions/${sid}/review`),
  demo: () => call('POST', '/demo'),
}
