// Every player has their own voice. With an ElevenLabs key the backend returns natural human
// voices; lines are fetched ahead of time so the next speaker starts the moment the last one ends,
// like a real call. Without a key we pick the most natural voices the browser has (Edge's
// "Natural" neural voices, Google voices in Chrome) and keep pitch close to normal, because
// pitch-shifting is what makes browser speech sound robotic.

const FAST = new URLSearchParams(window.location.search).has('fast') // for automated tests
const FEMALE = new Set(['Ava', 'Cleo'])
const BROWSER_RATE = { Narrator: 0.97, Ava: 1.04, Ben: 1.0, Cleo: 1.08, Dev: 0.98, Eli: 0.92 }
const FEMALE_HINTS = /(female|aria|jenny|sonia|libby|natasha|emma|ava|michelle|sara|serena|samantha|karen|moira|tessa|zira|hazel|susan|fiona|victoria|kate|google uk english female|google us english)/i
const MALE_HINTS = /(male|guy|ryan|thomas|william|christopher|eric|andrew|brian|davis|tony|daniel|arthur|oliver|alex|fred|george|rishi|aaron|google uk english male)/i

let audioCtx = null
export function audioContext() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)()
  return audioCtx
}

export function unlockAudio() {
  audioContext().resume()
  if ('speechSynthesis' in window) window.speechSynthesis.getVoices()
}

let status = null
async function backendHasVoices() {
  if (status === null) {
    status = await fetch('/api/health').then((r) => r.json()).catch(() => ({}))
  }
  return Boolean(status.elevenlabs)
}

const cache = new Map()
function fetchAudio(speaker, text) {
  const key = `${speaker}|${text}`
  if (!cache.has(key)) {
    cache.set(key, fetch('/api/tts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, speaker }),
    }).then((res) => (res.status === 200 ? res.blob() : null)).catch(() => null))
  }
  return cache.get(key)
}

// Start generating audio for upcoming lines while the current one plays.
export async function prefetch(lines) {
  if (FAST || !(await backendHasVoices())) return
  for (const l of lines) fetchAudio(l.speaker, l.text)
}

function rankVoice(v) {
  let score = 0
  if (/natural|neural|online/i.test(v.name)) score += 10
  if (/google/i.test(v.name)) score += 6
  if (/en-GB/i.test(v.lang)) score += 2
  if (v.localService === false) score += 1
  return score
}

const assigned = {}
function browserVoice(speaker) {
  if (assigned[speaker]) return assigned[speaker]
  const all = (window.speechSynthesis?.getVoices() ?? []).filter((v) => v.lang?.startsWith('en'))
  if (!all.length) return null
  const wantFemale = FEMALE.has(speaker)
  const fits = all.filter((v) => (wantFemale ? FEMALE_HINTS.test(v.name) && !/\bmale\b/i.test(v.name) : MALE_HINTS.test(v.name)))
  const pool = (fits.length ? fits : all).sort((a, b) => rankVoice(b) - rankVoice(a))
  const used = new Set(Object.values(assigned).map((v) => v.name))
  const voice = pool.find((v) => !used.has(v.name)) ?? pool[0]
  assigned[speaker] = voice
  return voice
}

async function playMp3(blob, onLevel) {
  const audio = new Audio(URL.createObjectURL(blob))
  const c = audioContext()
  const src = c.createMediaElementSource(audio)
  const an = c.createAnalyser()
  an.fftSize = 512
  src.connect(an)
  an.connect(c.destination)
  const buf = new Float32Array(an.fftSize)
  let raf
  const meter = () => {
    an.getFloatTimeDomainData(buf)
    let rms = 0
    for (let i = 0; i < buf.length; i++) rms += buf[i] ** 2
    onLevel?.(Math.min(1, Math.sqrt(rms / buf.length) * 5))
    raf = requestAnimationFrame(meter)
  }
  await c.resume()
  await audio.play()
  meter()
  await new Promise((r) => {
    audio.onended = r
    audio.onerror = r
  })
  cancelAnimationFrame(raf)
  onLevel?.(0)
}

function browserSpeak(speaker, text, onLevel) {
  return new Promise((resolve) => {
    if (!('speechSynthesis' in window)) return resolve()
    const u = new SpeechSynthesisUtterance(text)
    const voice = browserVoice(speaker)
    if (voice) u.voice = voice
    u.pitch = 1
    u.rate = BROWSER_RATE[speaker] ?? 1
    // No audio tap on speechSynthesis, so animate from word boundaries.
    let level = 0
    const pulse = setInterval(() => {
      level = Math.max(0, level * 0.7 + Math.random() * 0.35)
      onLevel?.(level)
    }, 60)
    u.onboundary = () => (level = 0.9)
    let finished = false
    const done = () => {
      if (finished) return
      finished = true
      clearInterval(pulse)
      onLevel?.(0)
      resolve()
    }
    u.onend = done
    u.onerror = done
    setTimeout(done, 2500 + text.length * 110) // some browsers never fire onend
    window.speechSynthesis.speak(u)
  })
}

export async function say(speaker, text, onLevel) {
  if (FAST) return new Promise((r) => setTimeout(r, 60))
  if (await backendHasVoices()) {
    const blob = await fetchAudio(speaker, text)
    cache.delete(`${speaker}|${text}`)
    if (blob) return playMp3(blob, onLevel)
  }
  return browserSpeak(speaker, text, onLevel)
}

export function usingHumanVoices() {
  return Boolean(status?.elevenlabs)
}
