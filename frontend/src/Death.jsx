import Face from './Face.jsx'

// Someone dies: the screen bleeds, their heartbeat runs down to a flatline, and the town learns
// who it was.
const DRIPS = [4, 11, 19, 27, 36, 44, 53, 61, 70, 78, 86, 94]

export default function Death({ who, how, role }) {
  const you = who === 'You'
  const line = how === 'vote'
    ? `${you ? 'were' : 'was'} voted out. ${you ? 'You were' : 'They were'} ${role === 'mafia' ? 'the MAFIA' : 'innocent'}.`
    : 'was found dead this morning.'
  return (
    <div className={`death ${how}`}>
      <div className="death-flash" />
      {DRIPS.map((x, i) => (
        <span key={x} className="drip" style={{ left: `${x}%`, '--d': `${(i * 137) % 900}ms`, '--h': `${18 + ((i * 53) % 42)}vh` }} />
      ))}
      <div className="death-body">
        <div className="death-face">
          <span><Face name={who} className="death-portrait" /></span>
          <span className="death-skull">💀</span>
          <svg className="death-crack" viewBox="0 0 100 100" aria-hidden="true">
            <path d="M52 0 L46 22 L58 34 L44 52 L56 66 L48 100" />
          </svg>
        </div>
        <svg className="ecg" viewBox="0 0 600 80" aria-hidden="true">
          <path d="M0 40 H70 l10 -26 l12 52 l10 -40 l8 14 H190 l10 -26 l12 52 l10 -40 l8 14 H320 l8 -16 l10 32 l8 -20 l6 4 H600" />
        </svg>
        <div className="death-name">{you ? 'YOU' : who.toUpperCase()}</div>
        <div className="death-line">{line}</div>
      </div>
    </div>
  )
}
