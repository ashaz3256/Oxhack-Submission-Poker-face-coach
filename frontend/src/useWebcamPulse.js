import { useEffect, useRef, useState } from 'react'

// Remote photoplethysmography: skin brightness in the green channel pulses with blood volume.
// We average a forehead patch, keep ~12 s of it, and take the strongest frequency in the
// 45-180 bpm band. Rough, but real, and it needs nothing but a webcam and steady light.
const FPS = 30
const WINDOW_S = 12
const ROI = { x: 0.4, y: 0.12, w: 0.2, h: 0.14 } // fraction of the frame; align your forehead to the box

function estimateBpm(buf) {
  if (buf.length < FPS * 6) return null
  const t0 = buf[0].t
  const t1 = buf[buf.length - 1].t
  const n = Math.floor((t1 - t0) * FPS)
  const xs = new Float64Array(n)
  let j = 0
  for (let i = 0; i < n; i++) {
    const t = t0 + i / FPS
    while (j < buf.length - 2 && buf[j + 1].t < t) j++
    const a = buf[j]
    const b = buf[j + 1]
    xs[i] = a.g + ((b.g - a.g) * (t - a.t)) / Math.max(b.t - a.t, 1e-6)
  }
  // Detrend with a 1 s moving average, then normalise.
  const out = new Float64Array(n)
  const half = Math.floor(FPS / 2)
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
      const w = (2 * Math.PI * f * i) / FPS
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

export function useWebcamPulse(enabled, onSample) {
  const videoRef = useRef(null)
  const [error, setError] = useState(null)
  const [bpm, setBpm] = useState(null)
  const cb = useRef(onSample)
  cb.current = onSample

  useEffect(() => {
    if (!enabled) return
    let stream
    let raf
    let timer
    const buf = []
    const recent = []
    const canvas = document.createElement('canvas')
    canvas.width = 160
    canvas.height = 120
    const ctx = canvas.getContext('2d', { willReadFrequently: true })

    const grab = () => {
      const v = videoRef.current
      if (v && v.readyState >= 2) {
        ctx.drawImage(v, 0, 0, canvas.width, canvas.height)
        const { data } = ctx.getImageData(
          Math.floor(ROI.x * canvas.width), Math.floor(ROI.y * canvas.height),
          Math.floor(ROI.w * canvas.width), Math.floor(ROI.h * canvas.height),
        )
        let g = 0
        for (let i = 1; i < data.length; i += 4) g += data[i]
        buf.push({ t: performance.now() / 1000, g: g / (data.length / 4) })
        const cutoff = buf[buf.length - 1].t - WINDOW_S
        while (buf.length && buf[0].t < cutoff) buf.shift()
      }
      raf = requestAnimationFrame(grab)
    }

    navigator.mediaDevices
      .getUserMedia({ video: { width: 640, height: 480 } })
      .then((s) => {
        stream = s
        videoRef.current.srcObject = s
        videoRef.current.play()
        raf = requestAnimationFrame(grab)
        timer = setInterval(() => {
          const est = estimateBpm(buf)
          if (!est) return
          recent.push(est)
          if (recent.length > 5) recent.shift()
          const median = [...recent].sort((a, b) => a - b)[Math.floor(recent.length / 2)]
          setBpm(Math.round(median))
          cb.current?.({ t: Date.now() / 1000, hr: Math.round(median * 10) / 10, br: null })
        }, 1000)
      })
      .catch((e) => setError(e.message))

    return () => {
      cancelAnimationFrame(raf)
      clearInterval(timer)
      stream?.getTracks().forEach((t) => t.stop())
    }
  }, [enabled])

  return { videoRef, bpm, error, roi: ROI }
}
