// Voice analysis on the microphone: loudness, pitch (autocorrelation) and voiced/silent frames.
// Per answer: median pitch and its wobble, loudness, how long before you started, the share
// of pauses, and (with the transcript) words per voiced second.

import { audioContext } from '../tts.js'

function autocorrPitch(buf, sampleRate) {
  // Normalised autocorrelation between 70 and 400 Hz.
  const n = buf.length
  let mean = 0
  for (let i = 0; i < n; i++) mean += buf[i]
  mean /= n
  const minLag = Math.floor(sampleRate / 400)
  const maxLag = Math.floor(sampleRate / 70)
  let bestLag = -1
  let best = 0
  let energy = 0
  for (let i = 0; i < n; i++) energy += (buf[i] - mean) ** 2
  if (energy === 0) return null
  for (let lag = minLag; lag <= maxLag; lag++) {
    let s = 0
    for (let i = 0; i < n - lag; i++) s += (buf[i] - mean) * (buf[i + lag] - mean)
    s /= energy
    if (s > best) {
      best = s
      bestLag = lag
    }
  }
  return best > 0.5 && bestLag > 0 ? sampleRate / bestLag : null
}

export class VoiceAnalyser {
  constructor(stream) {
    this.ctx = audioContext()
    const src = this.ctx.createMediaStreamSource(stream)
    this.analyser = this.ctx.createAnalyser()
    this.analyser.fftSize = 2048
    src.connect(this.analyser)
    this.buf = new Float32Array(this.analyser.fftSize)
    this.noise = 0.01
    this.live = { level: 0, pitch: null, voiced: false }
    this.turn = null
    this.lastVoiceAt = 0
    this.calib = []
    this.pauses = []
    this.timer = setInterval(() => this.tick(), 40)
  }

  tick() {
    this.analyser.getFloatTimeDomainData(this.buf)
    let rms = 0
    for (let i = 0; i < this.buf.length; i++) rms += this.buf[i] ** 2
    rms = Math.sqrt(rms / this.buf.length)
    // Track the room's noise floor while nobody is speaking.
    if (!this.turn) {
      this.calib.push(rms)
      if (this.calib.length > 100) this.calib.shift()
      const sorted = [...this.calib].sort((a, b) => a - b)
      this.noise = Math.max(0.004, sorted[Math.floor(sorted.length * 0.5)] || 0.01)
    }
    const voiced = rms > this.noise * 2.5
    const pitch = voiced ? autocorrPitch(this.buf, this.ctx.sampleRate) : null
    this.live = { level: Math.min(1, rms / 0.15), pitch: pitch ? Math.round(pitch) : this.live.pitch, voiced }
    const now = performance.now()
    if (voiced) this.lastVoiceAt = now
    if (this.turn) {
      const t = this.turn
      t.frames.push({ t: now, voiced, rms })
      if (pitch) t.pitches.push(pitch)
      if (voiced && t.firstVoiced === null) t.firstVoiced = now
    }
  }

  beginTurn() {
    this.ctx.resume()
    this.turn = { start: performance.now(), frames: [], pitches: [], firstVoiced: null }
  }

  endTurn(transcript) {
    const t = this.turn
    this.turn = null
    if (!t || t.firstVoiced === null) return { latency: null }
    const voicedFrames = t.frames.filter((f) => f.voiced)
    const first = t.frames.findIndex((f) => f.voiced)
    let last = t.frames.length - 1
    while (last > first && !t.frames[last].voiced) last--
    const span = t.frames.slice(first, last + 1)
    const voicedSeconds = (voicedFrames.length * 40) / 1000
    // Longest pause inside the answer: after the warm-up this sets how long a silence ends your turn.
    let gap = 0
    let longest = 0
    for (const f of span) {
      gap = f.voiced ? 0 : gap + 40
      longest = Math.max(longest, gap)
    }
    this.pauses.push(longest)
    const words = transcript.trim() ? transcript.trim().split(/\s+/).length : 0
    const p = [...t.pitches].sort((a, b) => a - b)
    const median = p.length ? p[Math.floor(p.length / 2)] : null
    const mean = p.length ? p.reduce((s, v) => s + v, 0) / p.length : null
    const sd = p.length > 1 ? Math.sqrt(p.reduce((s, v) => s + (v - mean) ** 2, 0) / (p.length - 1)) : null
    return {
      pitch: median,
      pitch_var: sd,
      loudness: voicedFrames.length ? voicedFrames.reduce((s, f) => s + f.rms, 0) / voicedFrames.length : null,
      rate: voicedSeconds > 0.3 ? words / voicedSeconds : null,
      pause: span.length ? 1 - span.filter((f) => f.voiced).length / span.length : null,
      latency: (t.firstVoiced - t.start) / 1000,
    }
  }

  // Your natural mid-answer pause plus a margin, between 0.9 and 1.6 seconds; null until we know you.
  personalSilenceMs() {
    if (this.pauses.length < 3) return null
    const sorted = [...this.pauses].sort((a, b) => a - b)
    const typical = sorted[Math.floor(sorted.length * 0.75)]
    return Math.max(900, Math.min(1600, typical + 350))
  }

  silentFor() {
    return performance.now() - this.lastVoiceAt
  }
}
