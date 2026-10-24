// Records one answer from the mic as a 16 kHz mono WAV, for the backup speech-to-text.

import { audioContext } from '../tts.js'

const RATE = 16000

export class WavRecorder {
  constructor(stream) {
    const c = audioContext()
    this.rate = c.sampleRate
    this.chunks = null
    const src = c.createMediaStreamSource(stream)
    this.node = c.createScriptProcessor(4096, 1, 1)
    this.node.onaudioprocess = (e) => {
      if (this.chunks) this.chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)))
    }
    const mute = c.createGain()
    mute.gain.value = 0
    src.connect(this.node)
    this.node.connect(mute)
    mute.connect(c.destination)
  }

  start() {
    this.chunks = []
  }

  stop() {
    const chunks = this.chunks ?? []
    this.chunks = null
    let n = 0
    for (const c of chunks) n += c.length
    const all = new Float32Array(n)
    let o = 0
    for (const c of chunks) {
      all.set(c, o)
      o += c.length
    }
    return toWav(downsample(all, this.rate, RATE), RATE)
  }
}

function downsample(buf, from, to) {
  if (from === to) return buf
  const ratio = from / to
  const out = new Float32Array(Math.floor(buf.length / ratio))
  for (let i = 0; i < out.length; i++) {
    const start = Math.floor(i * ratio)
    const end = Math.min(buf.length, Math.floor((i + 1) * ratio))
    let sum = 0
    for (let j = start; j < end; j++) sum += buf[j]
    out[i] = sum / Math.max(1, end - start)
  }
  return out
}

function toWav(samples, rate) {
  const view = new DataView(new ArrayBuffer(44 + samples.length * 2))
  const str = (o, s) => [...s].forEach((ch, i) => view.setUint8(o + i, ch.charCodeAt(0)))
  str(0, 'RIFF')
  view.setUint32(4, 36 + samples.length * 2, true)
  str(8, 'WAVE')
  str(12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true)
  view.setUint16(22, 1, true)
  view.setUint32(24, rate, true)
  view.setUint32(28, rate * 2, true)
  view.setUint16(32, 2, true)
  view.setUint16(34, 16, true)
  str(36, 'data')
  view.setUint32(40, samples.length * 2, true)
  for (let i = 0; i < samples.length; i++) {
    const v = Math.max(-1, Math.min(1, samples[i]))
    view.setInt16(44 + i * 2, v < 0 ? v * 0x8000 : v * 0x7fff, true)
  }
  return new Blob([view], { type: 'audio/wav' })
}
