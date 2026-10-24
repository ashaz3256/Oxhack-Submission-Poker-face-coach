// Every player has their own voice: ElevenLabs when the backend has a key, otherwise a distinct
// browser voice with its own pitch and speed. `onLevel` drives the talking animation.

const BROWSER = {
  Narrator: { pitch: 0.9, rate: 0.95, pick: 0 },
  Ava: { pitch: 1.15, rate: 1.05, pick: 1 },
  Ben: { pitch: 0.75, rate: 1.0, pick: 2 },
  Cleo: { pitch: 1.35, rate: 1.12, pick: 3 },
  Dev: { pitch: 1.0, rate: 0.95, pick: 4 },
  Eli: { pitch: 0.6, rate: 0.85, pick: 5 },
}

let audioCtx = null
function ctx() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)()
  return audioCtx
}

export function unlockAudio() {
  ctx().resume()
  if ('speechSynthesis' in window) window.speechSynthesis.getVoices()
}

function englishVoices() {
  const all = window.speechSynthesis?.getVoices() ?? []
  const en = all.filter((v) => v.lang?.startsWith('en'))
  return en.length ? en : all
}

async function playMp3(blob, onLevel) {
  const audio = new Audio(URL.createObjectURL(blob))
  const c = ctx()
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
    const cfg = BROWSER[speaker] ?? BROWSER.Narrator
    const voices = englishVoices()
    const u = new SpeechSynthesisUtterance(text)
    if (voices.length) u.voice = voices[cfg.pick % voices.length]
    u.pitch = cfg.pitch
    u.rate = cfg.rate
    // No audio tap on speechSynthesis, so animate from word boundaries.
    let level = 0
    const pulse = setInterval(() => {
      level = Math.max(0, level * 0.7 + Math.random() * 0.35)
      onLevel?.(level)
    }, 60)
    u.onboundary = () => (level = 0.9)
    const done = () => {
      clearInterval(pulse)
      onLevel?.(0)
      resolve()
    }
    u.onend = done
    u.onerror = done
    // Safety net: some browsers never fire onend.
    setTimeout(done, 1500 + text.length * 90)
    window.speechSynthesis.speak(u)
  })
}

const FAST = new URLSearchParams(window.location.search).has('fast') // for automated tests

export async function say(speaker, text, onLevel) {
  if (FAST) return new Promise((r) => setTimeout(r, 60))
  try {
    const res = await fetch('/api/tts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, speaker }),
    })
    if (res.status === 200) return await playMp3(await res.blob(), onLevel)
  } catch {
    // fall through to the browser voice
  }
  return browserSpeak(speaker, text, onLevel)
}
