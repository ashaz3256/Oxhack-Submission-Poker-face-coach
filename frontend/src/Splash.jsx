// The opening shot: an eye opens in the dark, looks around, locks on to you, and the title glitches
// in. Click or press any key to skip.
import { useEffect } from 'react'

export const SPLASH_MS = 5200

export default function Splash({ onDone }) {
  useEffect(() => {
    const t = setTimeout(onDone, SPLASH_MS)
    const skip = () => onDone()
    window.addEventListener('keydown', skip)
    return () => {
      clearTimeout(t)
      window.removeEventListener('keydown', skip)
    }
  }, [onDone])
  return (
    <div className="splash" onClick={onDone}>
      <div className="splash-grain" />
      <svg className="eye" viewBox="0 0 400 200" aria-hidden="true">
        <defs>
          <clipPath id="eye-shape">
            <path d="M10 100 Q200 -40 390 100 Q200 240 10 100 Z" />
          </clipPath>
          <radialGradient id="iris" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#ffb199" />
            <stop offset="35%" stopColor="#e5544b" />
            <stop offset="80%" stopColor="#6d1610" />
            <stop offset="100%" stopColor="#2a0606" />
          </radialGradient>
        </defs>
        <g clipPath="url(#eye-shape)">
          <rect width="400" height="200" fill="#e9e1dc" />
          <path className="vein" d="M10 100 Q60 90 110 104 T150 96" />
          <path className="vein" d="M390 100 Q340 112 300 96 T252 106" />
          <path className="vein" d="M40 70 Q90 84 128 78" />
          <path className="vein" d="M360 130 Q310 118 276 124" />
          <g className="iris">
            <circle cx="200" cy="100" r="62" fill="url(#iris)" />
            <circle className="pupil" cx="200" cy="100" r="24" fill="#050202" />
            <circle cx="218" cy="82" r="9" fill="#fff" opacity=".85" />
          </g>
          <rect className="lid top" x="0" y="-100" width="400" height="200" fill="#070505" />
          <rect className="lid bottom" x="0" y="100" width="400" height="200" fill="#070505" />
        </g>
        <path className="eye-outline" d="M10 100 Q200 -40 390 100 Q200 240 10 100 Z" />
      </svg>
      <div className="splash-title" data-text="TILT">TILT</div>
      <div className="splash-sub">your body is leaking information</div>
      <div className="splash-skip">click to skip</div>
    </div>
  )
}
