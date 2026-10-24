// Presage SmartSpectra sidecar: reads vitals from the camera and posts them to the Tilt backend.
//
//   cd presage && npm install
//   PRESAGE_API_KEY=... node presage.mjs            (or put the key in backend/.env)
//
// Sends pulse, breathing, HRV (RMSSD), the EDA proxy, blinking and talking to POST /api/presage
// about once a second. The game uses them for each answer instead of the rougher in-browser pulse.
// --mock sends clearly fake numbers (source "mock") to test the plumbing without a camera or key.
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const API = process.env.TILT_API || 'http://localhost:8000'
const here = path.dirname(fileURLToPath(import.meta.url))

function keyFromEnvFile() {
  try {
    const text = fs.readFileSync(path.join(here, '..', 'backend', '.env'), 'utf8')
    return text.match(/^PRESAGE_API_KEY=(.+)$/m)?.[1]?.trim() || null
  } catch {
    return null
  }
}

const last = (arr) => (Array.isArray(arr) && arr.length ? arr[arr.length - 1] : null)
const num = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : null)

// Pull the newest value of each signal out of one decoded Metrics message.
export function summarise(m) {
  const pulse = last(m?.cardio?.pulseRate)
  const br = last(m?.breathing?.rate)
  const hrv = last(m?.cardio?.hrv)
  const eda = last(m?.eda?.trace)
  const blink = last(m?.face?.blinking)
  const talk = last(m?.face?.talking)
  const out = {
    pulse: num(pulse?.value), pulse_conf: num(pulse?.confidence), pulse_stable: pulse?.stable ?? null,
    br: num(br?.value), br_conf: num(br?.confidence),
    hrv_rmssd: hrv?.stable === false ? null : num(hrv?.rmssd),
    eda: num(eda?.value),
    blinking: blink ? Boolean(blink.detected) : null,
    talking: talk ? Boolean(talk.detected) : null,
  }
  return Object.values(out).some((v) => v !== null) ? out : null
}

async function post(sample) {
  try {
    await fetch(`${API}/api/presage`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ samples: [sample] }),
    })
  } catch (e) {
    console.error('backend not reachable:', e.message)
  }
}

async function mock() {
  console.log('MOCK mode: sending fake vitals tagged source="mock". Not real measurements.')
  let t = 0
  setInterval(() => {
    t++
    post({ t: Date.now() / 1000, source: 'mock', pulse: 70 + 6 * Math.sin(t / 7), pulse_conf: 0.9, br: 15,
      br_conf: 0.9, hrv_rmssd: 40, eda: null, blinking: t % 4 === 0, talking: false })
  }, 1000)
}

async function live() {
  const key = process.env.PRESAGE_API_KEY || keyFromEnvFile()
  if (!key) {
    console.error('Set PRESAGE_API_KEY (env or backend/.env). Get one at https://physiology.presagetech.com')
    process.exit(1)
  }
  const { SmartSpectraSDK, CameraSelection, breathingMetrics, cardioMetrics, faceMetrics, edaMetrics, ValidationCode } =
    await import('@smartspectra/node-sdk')
  const { decodeMetrics } = await import('@smartspectra/node-sdk/messages')
  const hints = Object.fromEntries(Object.entries(ValidationCode).map(([k, v]) => [v, k]))

  const sdk = new SmartSpectraSDK({
    apiKey: key,
    requestedMetrics: [...breathingMetrics, ...cardioMetrics, ...faceMetrics, ...edaMetrics],
  })
  let lastSent = 0
  let lastHint = null
  sdk.on('processingStatus', (s) => console.log('status', s))
  sdk.on('validationStatus', (code, ts, hint) => {
    if (code !== lastHint) console.log('camera check:', hints[code] || code, hint || '')
    lastHint = code
  })
  sdk.on('error', (code, msg, retryable) => console.error('Presage error', code, msg, retryable ? '(retrying)' : ''))
  sdk.on('metrics', (buf) => {
    const s = summarise(decodeMetrics(buf))
    const now = Date.now() / 1000
    if (!s || now - lastSent < 0.9) return
    lastSent = now
    post({ t: now, source: 'presage', ...s })
    if (s.pulse) process.stdout.write(`\rpulse ${s.pulse.toFixed(0)}  breathing ${s.br?.toFixed(1) ?? '-'}  hrv ${s.hrv_rmssd?.toFixed(0) ?? '-'}   `)
  })

  const cams = SmartSpectraSDK.availableCameras()
  const pick = process.env.PRESAGE_CAMERA
  sdk.useCamera(pick ? CameraSelection.byId(pick) : CameraSelection.default)
  console.log('cameras:', cams.map((c) => `${c.id} (${c.name ?? ''})`).join(', ') || 'none found')
  sdk.start()
  console.log('Presage running. Look at the camera, keep still, steady light. Ctrl+C to stop.')
  process.on('SIGINT', async () => {
    await sdk.stopAsync()
    await sdk.destroy()
    process.exit(0)
  })
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  ;(process.argv.includes('--mock') ? mock() : live()).catch((e) => {
    console.error(e.message)
    process.exit(1)
  })
}
