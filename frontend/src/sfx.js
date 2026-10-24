// Small synthesised sound effects for the creepy bits: a low drone when night falls, heartbeats
// that stop on a flatline when someone dies, and a sting for the game's start. No audio files.

import { audioContext } from './tts.js'

const FAST = new URLSearchParams(window.location.search).has('fast')

function envelope(c, gain, at, attack, hold, release, peak) {
  gain.gain.setValueAtTime(0.0001, at)
  gain.gain.exponentialRampToValueAtTime(peak, at + attack)
  gain.gain.setValueAtTime(peak, at + attack + hold)
  gain.gain.exponentialRampToValueAtTime(0.0001, at + attack + hold + release)
}

function tone(type, freq, at, { attack = 0.01, hold = 0.05, release = 0.2, peak = 0.3, to = null, filter = null } = {}) {
  const c = audioContext()
  const osc = c.createOscillator()
  const gain = c.createGain()
  osc.type = type
  osc.frequency.setValueAtTime(freq, at)
  if (to) osc.frequency.exponentialRampToValueAtTime(to, at + attack + hold + release)
  let node = osc
  if (filter) {
    const f = c.createBiquadFilter()
    f.type = 'lowpass'
    f.frequency.value = filter
    osc.connect(f)
    node = f
  }
  node.connect(gain)
  gain.connect(c.destination)
  envelope(c, gain, at, attack, hold, release, peak)
  osc.start(at)
  osc.stop(at + attack + hold + release + 0.05)
}

function heartbeat(at) {
  tone('sine', 62, at, { attack: 0.005, hold: 0.04, release: 0.16, peak: 0.6, to: 40 })
  tone('sine', 55, at + 0.22, { attack: 0.005, hold: 0.03, release: 0.14, peak: 0.45, to: 38 })
}

export function playDeath() {
  if (FAST) return
  const c = audioContext()
  const t = c.currentTime + 0.05
  // Heart racing, slowing, then the flatline.
  ;[0, 0.55, 1.1, 1.75, 2.5].forEach((d) => heartbeat(t + d))
  tone('sine', 880, t + 3.0, { attack: 0.02, hold: 1.6, release: 0.4, peak: 0.12 })
  // A low scrape underneath.
  tone('sawtooth', 70, t, { attack: 0.4, hold: 1.8, release: 1.2, peak: 0.08, to: 45, filter: 400 })
}

export function playNight() {
  if (FAST) return
  const c = audioContext()
  const t = c.currentTime + 0.05
  tone('sawtooth', 55, t, { attack: 1.2, hold: 1.5, release: 2.0, peak: 0.07, filter: 300 })
  tone('sine', 110, t + 0.3, { attack: 1.0, hold: 1.2, release: 1.8, peak: 0.05, to: 98 })
}

export function playIntro() {
  if (FAST) return
  const c = audioContext()
  const t = c.currentTime + 0.05
  tone('sawtooth', 41, t, { attack: 1.5, hold: 1.8, release: 1.5, peak: 0.09, filter: 260 })
  tone('triangle', 220, t + 1.4, { attack: 0.01, hold: 0.1, release: 1.6, peak: 0.12, to: 207 })
  tone('triangle', 233, t + 1.4, { attack: 0.01, hold: 0.1, release: 1.6, peak: 0.1, to: 220 })
  heartbeat(t + 2.6)
  heartbeat(t + 3.4)
}

export function playVerdict(won) {
  if (FAST) return
  const c = audioContext()
  const t = c.currentTime + 0.05
  const notes = won ? [262, 330, 392, 523] : [294, 277, 262, 208]
  notes.forEach((f, i) => tone('triangle', f, t + i * 0.18, { attack: 0.01, hold: 0.12, release: 0.5, peak: 0.15 }))
}

// A phone ringing out (UK double ring), then little blips as people join the call.
export function playRing() {
  if (FAST) return
  const c = audioContext()
  const t = c.currentTime + 0.05
  for (const start of [0, 0.6, 2.0, 2.6]) {
    tone('sine', 400, t + start, { attack: 0.02, hold: 0.36, release: 0.05, peak: 0.08 })
    tone('sine', 450, t + start, { attack: 0.02, hold: 0.36, release: 0.05, peak: 0.08 })
  }
}

export function playJoin(i = 0) {
  if (FAST) return
  const c = audioContext()
  const t = c.currentTime + 0.02
  tone('triangle', 520 + i * 60, t, { attack: 0.01, hold: 0.05, release: 0.18, peak: 0.12 })
  tone('triangle', 780 + i * 60, t + 0.08, { attack: 0.01, hold: 0.05, release: 0.2, peak: 0.1 })
}
