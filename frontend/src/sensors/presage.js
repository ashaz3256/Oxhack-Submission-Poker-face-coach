// Presage vitals: records the webcam in rolling clips and posts each one to the backend, which
// sends it to the Presage Physiology API for heart rate and breathing rate. Only runs when the
// backend has a PRESAGE_API_KEY. Readings arrive a little after each clip.

const CLIP_MS = 15000

function pickType() {
  const types = ['video/mp4;codecs=avc1', 'video/mp4', 'video/webm;codecs=vp9', 'video/webm']
  return types.find((t) => window.MediaRecorder?.isTypeSupported?.(t)) ?? ''
}

export class PresageRecorder {
  constructor(stream, sessionId) {
    this.stream = new MediaStream(stream.getVideoTracks())
    this.sid = sessionId
    this.type = pickType()
    this.ext = this.type.startsWith('video/mp4') ? 'mp4' : 'webm'
    this.live = { enabled: false, hr: null, rr: null, status: 'off' }
    this.paused = false
    this.stopped = false
  }

  async start() {
    const health = await fetch('/api/health').then((r) => r.json()).catch(() => ({}))
    if (!health.presage || !window.MediaRecorder) return false
    this.live = { ...this.live, enabled: true, status: 'waiting' }
    this.loop()
    this.poll = setInterval(() => this.refresh(), 3000)
    return true
  }

  async loop() {
    while (!this.stopped) {
      if (this.paused) {
        await new Promise((r) => setTimeout(r, 500))
        continue
      }
      const blob = await this.record()
      if (blob && blob.size > 0 && !this.paused) {
        fetch(`/api/sessions/${this.sid}/vitals?ext=${this.ext}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/octet-stream' },
          body: blob,
        }).catch(() => {})
      }
    }
  }

  record() {
    return new Promise((resolve) => {
      let rec
      try {
        rec = new MediaRecorder(this.stream, this.type ? { mimeType: this.type, videoBitsPerSecond: 2_500_000 } : undefined)
      } catch {
        this.stopped = true
        return resolve(null)
      }
      const chunks = []
      rec.ondataavailable = (e) => e.data.size && chunks.push(e.data)
      rec.onstop = () => resolve(new Blob(chunks, { type: this.type || 'video/webm' }))
      rec.start()
      setTimeout(() => rec.state !== 'inactive' && rec.stop(), CLIP_MS)
    })
  }

  async refresh() {
    const v = await fetch(`/api/sessions/${this.sid}/vitals`).then((r) => r.json()).catch(() => null)
    if (v) this.live = v
  }

  stop() {
    this.stopped = true
    clearInterval(this.poll)
  }
}
