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
  if ('speechSynthesis' in window) {
    // Chrome can carry a stuck speech queue over from a reload; clearing it inside a click fixes that.
    window.speechSynthesis.cancel()
    window.speechSynthesis.getVoices()
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(''))
  }
}

let status = null
let failures = 0
let humanOff = false     // ElevenLabs kept failing, so this session uses browser voices
let voiceError = null
async function backendHasVoices() {
  if (status === null) {
    status = await fetch('/api/health').then((r) => r.json()).catch(() => ({}))
    voiceError = status.voice_error ?? null
  }
  return Boolean(status.elevenlabs) && !humanOff
}

function noteFailure(reason) {
  voiceError = reason || voiceError || 'ElevenLabs gave no audio'
  failures += 1
  if (failures >= 2) humanOff = true
  console.warn('Human voice failed, using the browser voice instead:', voiceError)
}

const cache = new Map()
function fetchAudio(speaker, text) {
  const key = `${speaker}|${text}`
  if (!cache.has(key)) {
    const abort = new AbortController()
    const timer = setTimeout(() => abort.abort(), 15000)
    cache.set(key, fetch('/api/tts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, speaker }),
      signal: abort.signal,
    }).then((res) => {
      if (res.status === 200) return res.blob()
      noteFailure(res.headers.get('X-TTS-Error'))
      return null
    }).catch(() => {
      noteFailure('ElevenLabs took too long to answer')
      return null
    }).finally(() => clearTimeout(timer)))
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

// The pause button freezes whoever is talking mid-sentence.
let current = null
let paused = false
let endCurrent = null   // finishes whatever line is playing, for the skip button
let skipRequested = false
export function pauseSpeech() {
  paused = true
  current?.pause()
  window.speechSynthesis?.pause()
}
export function resumeSpeech() {
  paused = false
  current?.play().catch(() => {})
  window.speechSynthesis?.resume()
}

// Plays straight to the speakers (not through Web Audio, so a suspended audio context or an odd
// output device can't silence it). The talking animation taps a copy of the stream.
async function playMp3(blob, onLevel) {
  const audio = new Audio(URL.createObjectURL(blob))
  current = audio
  try {
    await audio.play()
  } catch (e) {
    current = null
    noteFailure(`The browser blocked audio playback (${e.name})`)
    return false
  }
  failures = 0
  let raf
  let meter = null
  try {
    const stream = audio.captureStream?.() ?? audio.mozCaptureStream?.()
    if (stream?.getAudioTracks().length) {
      const c = audioContext()
      const an = c.createAnalyser()
      an.fftSize = 512
      c.createMediaStreamSource(stream).connect(an)
      const buf = new Float32Array(an.fftSize)
      meter = () => {
        an.getFloatTimeDomainData(buf)
        let rms = 0
        for (let i = 0; i < buf.length; i++) rms += buf[i] ** 2
        onLevel?.(Math.min(1, Math.sqrt(rms / buf.length) * 5))
        raf = requestAnimationFrame(meter)
      }
    }
  } catch {
    meter = null
  }
  if (!meter) {
    let level = 0
    meter = () => {
      level = audio.paused ? 0 : Math.max(0, level * 0.7 + Math.random() * 0.35)
      onLevel?.(level)
      raf = setTimeout(meter, 60)
    }
  }
  meter()
  await new Promise((r) => {
    audio.onended = r
    audio.onerror = r
    endCurrent = r
    if (skipRequested) r()
  })
  endCurrent = null
  audio.pause()
  cancelAnimationFrame(raf)
  clearTimeout(raf)
  current = null
  onLevel?.(0)
  return true
}

function browserSpeak(speaker, text, onLevel) {
  return new Promise((resolve) => {
    if (!('speechSynthesis' in window)) return resolve()
    const u = new SpeechSynthesisUtterance(cleanText(text))
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
    // Some browsers never fire onend, so give up after a generous time spent actually talking.
    let budget = 2500 + text.length * 110
    const guard = setInterval(() => {
      if (!paused) budget -= 250
      if (budget <= 0) done()
    }, 250)
    let finished = false
    endCurrent = () => {
      window.speechSynthesis.cancel()
      done()
    }
    function done() {
      if (finished) return
      finished = true
      endCurrent = null
      clearInterval(pulse)
      clearInterval(guard)
      onLevel?.(0)
      resolve()
    }
    u.onend = done
    u.onerror = done
    if (window.speechSynthesis.speaking && !paused) window.speechSynthesis.cancel()
    window.speechSynthesis.speak(u)
  })
}

// Delivery tags like [whispers] are for the voice model, not for captions or browser voices.
export const cleanText = (text) => (text || '').replace(/\[[a-zA-Z ]{2,24}\]\s*/g, '').trim()

export async function say(speaker, text, onLevel) {
  if (FAST) return new Promise((r) => setTimeout(r, 60))
  skipRequested = false
  if (await backendHasVoices()) {
    const blob = await fetchAudio(speaker, text)
    cache.delete(`${speaker}|${text}`)
    if (skipRequested) return
    if (blob && (await playMp3(blob, onLevel))) return
  }
  if (skipRequested) return
  return browserSpeak(speaker, text, onLevel)
}

export function usingHumanVoices() {
  return Boolean(status?.elevenlabs) && !humanOff
}

// For the sidebar: which voices are playing, and why ElevenLabs isn't if it isn't.
export function voiceStatus() {
  return { checked: status !== null, human: usingHumanVoices(), keySet: Boolean(status?.elevenlabs), error: voiceError }
}

// Cut the current line short and move on to the next one.
export function skipSpeech() {
  skipRequested = true   // also covers a line that is still loading or just starting
  endCurrent?.()
}
