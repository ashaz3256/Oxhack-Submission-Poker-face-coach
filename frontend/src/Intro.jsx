// The cold open before each game: night over a little town, a blood moon, fog and bats.
const BATS = [0, 1, 2, 3, 4]

export default function Intro({ game }) {
  return (
    <div className="intro">
      <div className="intro-stars" />
      <div className="intro-moon" />
      {BATS.map((i) => (
        <span key={i} className="intro-bat" style={{ '--i': i }}>🦇</span>
      ))}
      <svg className="intro-town" viewBox="0 0 1200 220" preserveAspectRatio="none" aria-hidden="true">
        <path d="M0 220 V150 h60 v-30 l25 -25 l25 25 v30 h40 v-60 h30 v-20 h20 v20 h30 v80 h50 v-40 l35 -35 l35 35 v40
          h40 v-90 l12 -30 l12 30 v90 h60 v-50 h70 v-25 l35 -30 l35 30 v75 h40 v-35 h50 v35 h30 v-70 l30 -40 l30 40 v70
          h55 v-45 l28 -28 l28 28 v45 h70 v-30 h60 v30 h40 V220 Z" />
        {[95, 300, 470, 640, 905, 1060].map((x, i) => (
          <rect key={x} className="intro-window" x={x} y={160 + (i % 2) * 14} width="9" height="12" style={{ animationDelay: `${i * 0.4}s` }} />
        ))}
      </svg>
      <div className="intro-fog" />
      <div className="intro-fog two" />
      <div className="intro-text">
        <div className="intro-small">Game {game}</div>
        <div className="intro-title">Night falls on Tilt</div>
        <div className="intro-sub">Six players. One liar. Every word is watched.</div>
      </div>
    </div>
  )
}
