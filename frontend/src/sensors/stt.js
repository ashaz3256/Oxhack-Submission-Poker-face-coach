// Speech to text with the browser's Web Speech API (Chrome and Edge). Hands-free: we open the
// mic for one turn and close it when you stop talking.

const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition
export const sttSupported = Boolean(Recognition)

// Errors that mean the browser's recognition will never work here (Brave, blocked network,
// mic permission). After one of these we switch to the backup: record and transcribe with Gemini.
const FATAL = new Set(['not-allowed', 'service-not-allowed', 'network', 'audio-capture', 'language-not-supported'])
export const sttState = { broken: !sttSupported, error: sttSupported ? null : 'This browser has no built-in speech recognition' }

export function markBroken(reason) {
  sttState.broken = true
  sttState.error = reason
}

// Backup turn-taking without recognition: wait until you've spoken and then gone quiet.
export function waitForSpeech({ maxMs = 20000, silenceMs = 1500, voice, control, onCountdown }) {
  return new Promise((resolve) => {
    const started = performance.now()
    let voicedMs = 0
    let done = false
    const finish = (override) => {
      if (done) return
      done = true
      clearInterval(timer)
      resolve(override ?? null)
    }
    if (control) control.stop = finish
    const timer = setInterval(() => {
      if (voice && voice.silentFor() < 120) voicedMs += 100
      const spoke = voicedMs >= 300
      onCountdown?.(spoke ? Math.min(1, voice.silentFor() / silenceMs) : 0)
      if (spoke && voice.silentFor() > silenceMs) finish()
      if (performance.now() - started > maxMs) finish()
    }, 100)
  })
}

// listen({ maxMs, silenceMs, voice, onInterim, control }) -> Promise<string>
// Ends when you have said something and then gone quiet for `silenceMs`, or at `maxMs`.
// `control.stop(text)` ends it early (a click, or the pause button) and resolves with `text`.
// `onCountdown(f)` reports how far through the closing silence you are, 0 to 1.
export function listen({ maxMs = 20000, silenceMs = 1500, voice, onInterim, control, onCountdown }) {
  return new Promise((resolve) => {
    let finalText = ''
    let interim = ''
    let lastResult = performance.now()
    let done = false
    const started = performance.now()
    const rec = new Recognition()
    rec.lang = 'en-GB'
    rec.continuous = true
    rec.interimResults = true

    rec.onresult = (e) => {
      interim = ''
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const r = e.results[i]
        if (r.isFinal) finalText += r[0].transcript + ' '
        else interim += r[0].transcript
      }
      lastResult = performance.now()
      onInterim?.((finalText + interim).trim())
    }
    rec.onend = () => {
      // Chrome stops on its own after long silences; keep going until we decide we're done.
      if (!done && !sttState.broken) {
        try {
          rec.start()
        } catch {
          // already started
        }
      }
    }
    rec.onerror = (e) => {
      if (FATAL.has(e.error)) {
        markBroken(`Browser speech recognition failed (${e.error})`)
        finish()
      }
    }

    const finish = (override) => {
      if (done) return
      done = true
      clearInterval(timer)
      try {
        rec.stop()
      } catch {
        // fine
      }
      resolve(override ?? (finalText + interim).trim())
    }
    if (control) control.stop = finish

    const timer = setInterval(() => {
      const now = performance.now()
      const said = (finalText + interim).trim().length > 0
      const quietMic = voice ? voice.silentFor() > silenceMs * 0.8 : true
      onCountdown?.(said ? Math.min(1, (now - lastResult) / silenceMs) : 0)
      if (said && now - lastResult > silenceMs && quietMic) finish()
      if (now - started > maxMs) finish()
    }, 150)

    try {
      rec.start()
    } catch (e) {
      markBroken(`Browser speech recognition failed (${e.name})`)
      finish()
    }
  })
}
