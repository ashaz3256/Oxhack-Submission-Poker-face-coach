// Face scan: MediaPipe Face Landmarker (478 points + 52 blendshapes) on the webcam, ~20 fps.
// Per answer we summarise blinks, gaze, brows, lips, smile and head movement, and we keep a
// webcam pulse estimate (rPPG) from the green channel of the forehead.
import { DrawingUtils, FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision'

const FPS_TARGET = 20
const PULSE_WINDOW_S = 12

function bs(categories) {
  const m = {}
  for (const c of categories) m[c.categoryName] = c.score
  return m
}

function estimateBpm(buf) {
  if (buf.length < 100) return null
  const t0 = buf[0].t
  const t1 = buf[buf.length - 1].t
  const fs = 20
  const n = Math.floor((t1 - t0) * fs)
  if (n < 120) return null
  const xs = new Float64Array(n)
  let j = 0
  for (let i = 0; i < n; i++) {
    const t = t0 + i / fs
    while (j < buf.length - 2 && buf[j + 1].t < t) j++
    const a = buf[j]
    const b = buf[j + 1]
    xs[i] = a.g + ((b.g - a.g) * (t - a.t)) / Math.max(b.t - a.t, 1e-6)
  }
  const out = new Float64Array(n)
  const half = 10
  for (let i = 0; i < n; i++) {
    let s = 0
    let c = 0
    for (let k = Math.max(0, i - half); k < Math.min(n, i + half); k++) {
      s += xs[k]
      c++
    }
    out[i] = xs[i] - s / c
  }
  let best = 0
  let bestF = 0
  for (let f = 0.75; f <= 3.0; f += 0.01) {
    let re = 0
    let im = 0
    for (let i = 0; i < n; i++) {
      const w = (2 * Math.PI * f * i) / fs
      re += out[i] * Math.cos(w)
      im += out[i] * Math.sin(w)
    }
    const p = re * re + im * im
    if (p > best) {
      best = p
      bestF = f
    }
  }
  return bestF * 60
}

export class FaceScanner {
  constructor() {
    this.landmarker = null
    this.live = { face: false, blink: 0, brow_raise: 0, brow_down: 0, lip_press: 0, gaze_away: 0, smile: 0, jaw: 0, hr: null }
    this.turn = null
    this.pulse = []
    this.hrHistory = []
    this.lastFrame = 0
    this.blinkClosed = false
    this.sampler = document.createElement('canvas')
    this.sampler.width = 32
    this.sampler.height = 32
    this.sctx = this.sampler.getContext('2d', { willReadFrequently: true })
  }

  async init() {
    const files = await FilesetResolver.forVisionTasks('/mediapipe')
    this.landmarker = await FaceLandmarker.createFromOptions(files, {
      baseOptions: { modelAssetPath: '/models/face_landmarker.task', delegate: 'GPU' },
      runningMode: 'VIDEO',
      numFaces: 1,
      outputFaceBlendshapes: true,
    })
  }

  start(video, canvas) {
    this.video = video
    this.canvas = canvas
    this.ctx = canvas.getContext('2d')
    this.draw = new DrawingUtils(this.ctx)
    this.running = true
    const loop = () => {
      if (!this.running) return
      const now = performance.now()
      if (now - this.lastFrame >= 1000 / FPS_TARGET && video.readyState >= 2 && this.landmarker) {
        this.lastFrame = now
        this.frame(now)
      }
      this.raf = requestAnimationFrame(loop)
    }
    this.raf = requestAnimationFrame(loop)
    this.hrTimer = setInterval(() => {
      const bpm = estimateBpm(this.pulse)
      if (bpm) {
        this.hrHistory.push(bpm)
        if (this.hrHistory.length > 5) this.hrHistory.shift()
        this.live.hr = Math.round([...this.hrHistory].sort((a, b) => a - b)[Math.floor(this.hrHistory.length / 2)])
      }
    }, 1000)
  }

  stop() {
    this.running = false
    cancelAnimationFrame(this.raf)
    clearInterval(this.hrTimer)
  }

  frame(now) {
    const { video, canvas, ctx } = this
    if (canvas.width !== video.videoWidth) {
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
    }
    const res = this.landmarker.detectForVideo(video, now)
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    const lm = res.faceLandmarks?.[0]
    if (!lm) {
      this.live.face = false
      return
    }
    this.live.face = true
    const b = bs(res.faceBlendshapes[0].categories)
    const blink = (b.eyeBlinkLeft + b.eyeBlinkRight) / 2
    const f = {
      blink,
      brow_raise: b.browInnerUp,
      brow_down: (b.browDownLeft + b.browDownRight) / 2,
      lip_press: (b.mouthPressLeft + b.mouthPressRight + b.mouthRollLower + b.mouthRollUpper) / 4,
      gaze_away: (Math.max(b.eyeLookOutLeft, b.eyeLookInLeft, b.eyeLookDownLeft, b.eyeLookUpLeft) +
        Math.max(b.eyeLookOutRight, b.eyeLookInRight, b.eyeLookDownRight, b.eyeLookUpRight)) / 2,
      smile: (b.mouthSmileLeft + b.mouthSmileRight) / 2,
      jaw: b.jawOpen,
    }
    Object.assign(this.live, f)

    // Blink = eyes closing past 0.5 after being open.
    let blinked = false
    if (blink > 0.5 && !this.blinkClosed) {
      this.blinkClosed = true
      blinked = true
    } else if (blink < 0.3) {
      this.blinkClosed = false
    }

    if (this.turn) {
      const t = this.turn
      t.frames++
      for (const k of ['brow_raise', 'brow_down', 'lip_press', 'gaze_away', 'smile']) t.sum[k] += f[k]
      if (blinked) t.blinks++
      t.nose.push([lm[1].x, lm[1].y])
    }

    // Pulse: mean green of a forehead patch, sized to the face.
    const w = video.videoWidth
    const h = video.videoHeight
    const faceW = Math.abs(lm[454].x - lm[234].x) * w
    const cx = lm[151].x * w
    const cy = lm[151].y * h
    const size = Math.max(8, faceW * 0.18)
    this.sctx.drawImage(video, cx - size / 2, cy - size / 2, size, size * 0.6, 0, 0, 32, 32)
    const data = this.sctx.getImageData(0, 0, 32, 32).data
    let g = 0
    for (let i = 1; i < data.length; i += 4) g += data[i]
    this.pulse.push({ t: now / 1000, g: g / (data.length / 4) })
    while (this.pulse.length && this.pulse[0].t < now / 1000 - PULSE_WINDOW_S) this.pulse.shift()

    // The scan overlay: eyes, brows, lips and face outline.
    const C = FaceLandmarker
    this.draw.drawConnectors(lm, C.FACE_LANDMARKS_TESSELATION, { color: 'rgba(224,176,74,0.10)', lineWidth: 0.5 })
    this.draw.drawConnectors(lm, C.FACE_LANDMARKS_FACE_OVAL, { color: 'rgba(224,176,74,0.7)', lineWidth: 1.5 })
    for (const conn of [C.FACE_LANDMARKS_LEFT_EYE, C.FACE_LANDMARKS_RIGHT_EYE, C.FACE_LANDMARKS_LEFT_IRIS, C.FACE_LANDMARKS_RIGHT_IRIS]) {
      this.draw.drawConnectors(lm, conn, { color: '#4aa3e0', lineWidth: 1.5 })
    }
    for (const conn of [C.FACE_LANDMARKS_LEFT_EYEBROW, C.FACE_LANDMARKS_RIGHT_EYEBROW]) {
      this.draw.drawConnectors(lm, conn, { color: '#5cc98a', lineWidth: 2 })
    }
    this.draw.drawConnectors(lm, C.FACE_LANDMARKS_LIPS, { color: '#e5544b', lineWidth: 2 })
    ctx.strokeStyle = 'rgba(255,255,255,0.6)'
    ctx.strokeRect(cx - size / 2, cy - size / 2, size, size * 0.6)
  }

  beginTurn() {
    this.turn = { start: performance.now(), frames: 0, blinks: 0, nose: [], sum: { brow_raise: 0, brow_down: 0, lip_press: 0, gaze_away: 0, smile: 0 } }
  }

  endTurn() {
    const t = this.turn
    this.turn = null
    if (!t || t.frames < 5) return { hr: this.live.hr }
    const minutes = (performance.now() - t.start) / 60000
    const mean = (k) => t.sum[k] / t.frames
    const xs = t.nose.map((p) => p[0])
    const ys = t.nose.map((p) => p[1])
    const sd = (a) => {
      const m = a.reduce((s, v) => s + v, 0) / a.length
      return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length)
    }
    return {
      blink: t.blinks / Math.max(minutes, 1 / 60),
      brow_raise: mean('brow_raise'),
      brow_down: mean('brow_down'),
      lip_press: mean('lip_press'),
      gaze_away: mean('gaze_away'),
      smile: mean('smile'),
      head_motion: (sd(xs) + sd(ys)) / 2,
      hr: this.live.hr,
    }
  }
}
