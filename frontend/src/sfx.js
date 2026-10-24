// Sound effects made once with ElevenLabs (backend/scripts/make_sfx.py) and served from /sfx.
// Every call is safe when a file is missing: it just stays silent.
import { audioContext } from './tts.js'

const NAMES = ['heartbeat', 'night', 'card_flip', 'deal', 'body_found', 'gavel', 'vote_tension', 'win', 'lose']
const buffers = {}
const loops = {}
const FAST = new URLSearchParams(window.location.search).has('fast')

export function preloadSfx() {
  if (FAST) return
  for (const name of NAMES) {
    if (buffers[name] !== undefined) continue
    buffers[name] = null
    fetch(`/sfx/${name}.mp3`)
      .then((r) => (r.ok && (r.headers.get('content-type') || '').includes('audio') ? r.arrayBuffer() : null))
      .then((b) => (b ? audioContext().decodeAudioData(b) : null))
      .then((buf) => (buffers[name] = buf))
      .catch(() => (buffers[name] = null))
  }
}

function source(name, volume, rate = 1, loop = false) {
  const buf = buffers[name]
  if (!buf) return null
  const c = audioContext()
  const src = c.createBufferSource()
  const gain = c.createGain()
  src.buffer = buf
  src.loop = loop
  src.playbackRate.value = rate
  gain.gain.value = volume
  src.connect(gain)
  gain.connect(c.destination)
  src.start()
  return { src, gain }
}

export function playSfx(name, volume = 0.6) {
  source(name, volume)
}

// Looping beds (night ambience, vote tension, heartbeat). One instance per name.
export function startLoop(name, volume = 0.3, rate = 1) {
  if (loops[name]) return setLoop(name, volume, rate)
  const s = source(name, 0, rate, true)
  if (!s) return
  s.gain.gain.linearRampToValueAtTime(volume, audioContext().currentTime + 0.6)
  loops[name] = s
}

export function setLoop(name, volume, rate) {
  const s = loops[name]
  if (!s) return
  const t = audioContext().currentTime
  if (volume != null) s.gain.gain.linearRampToValueAtTime(volume, t + 0.4)
  if (rate != null) s.src.playbackRate.linearRampToValueAtTime(rate, t + 0.4)
}

export function stopLoop(name) {
  const s = loops[name]
  if (!s) return
  delete loops[name]
  const t = audioContext().currentTime
  s.gain.gain.linearRampToValueAtTime(0, t + 0.6)
  s.src.stop(t + 0.7)
}

// The heartbeat follows how strongly the town's lie detector suspects you (0..1).
export function heartbeat(pLie) {
  const p = Math.max(0, Math.min(1, pLie ?? 0.4))
  startLoop('heartbeat', 0.15 + 0.5 * p, 0.85 + 0.9 * p)
}

// Cues keyed off what the narrator says, so the game logic does not need to know about sound.
const CUES = [
  [/^Six cards are face down/, () => playSfx('deal', 0.5)],
  [/^You drew card/, () => playSfx('card_flip', 0.8)],
  [/^Night falls/, () => { stopLoop('vote_tension'); startLoop('night', 0.25) }],
  [/found dead/, () => { stopLoop('night'); playSfx('body_found', 0.6) }],
  [/^Morning comes|^Day one/, () => stopLoop('night')],
  [/^Time to vote/, () => startLoop('vote_tension', 0.2)],
  [/eliminated\.|^The town eliminates/, () => { stopLoop('vote_tension'); playSfx('gavel', 0.7) }],
  [/You win!/, () => { stopLoop('night'); stopLoop('vote_tension'); playSfx('win', 0.6) }],
  [/You lose\./, () => { stopLoop('night'); stopLoop('vote_tension'); playSfx('lose', 0.6) }],
]

export function cueFor(speaker, text) {
  if (speaker !== 'Narrator') return
  for (const [re, fn] of CUES) if (re.test(text)) fn()
}
