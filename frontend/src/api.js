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
  create: () => call('POST', '/sessions', {}),
  calibrate: (sid, transcript, raw, isLie, question) => call('POST', `/sessions/${sid}/calibrate`, { transcript, raw, is_lie: isLie, question }),
  newGame: (sid) => call('POST', `/sessions/${sid}/games`),
  speak: (sid, transcript, raw) => call('POST', `/sessions/${sid}/speak`, { transcript, raw }),
  report: (sid) => call('GET', `/sessions/${sid}/report`),
  review: (sid) => call('POST', `/sessions/${sid}/review`),
  demo: () => call('POST', '/demo'),
}
