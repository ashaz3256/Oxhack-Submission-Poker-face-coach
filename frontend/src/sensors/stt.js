// Speech to text with the browser's Web Speech API (Chrome and Edge). Hands-free: we open the
// mic for one turn and close it when you stop talking.

const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition
export const sttSupported = Boolean(Recognition)

// listen({ maxMs, silenceMs, voice, onInterim, control }) -> Promise<string>
// Ends when you have said something and then gone quiet for `silenceMs`, or at `maxMs`.
// `control.stop(text)` ends it early (a click, or the pause button) and resolves with `text`.
export function listen({ maxMs = 20000, silenceMs = 1500, voice, onInterim, control }) {
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
      if (!done) {
        try {
          rec.start()
        } catch {
          // already started
        }
      }
    }
    rec.onerror = () => {}

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
      if (said && now - lastResult > silenceMs && quietMic) finish()
      if (now - started > maxMs) finish()
    }, 150)

    rec.start()
  })
}
