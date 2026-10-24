import { useState } from 'react'

// Each player's portrait (frontend/public/avatars/<name>.webp, made with ElevenLabs image
// generation to match their designed voice). Falls back to an emoji if the image is missing.
export const EMOJI = { Ava: '👩‍🦰', Ben: '🧔', Cleo: '👩‍🦱', Dev: '👨‍🦲', Eli: '🧓', Narrator: '🕯️', You: '😶' }
const PORTRAITS = new Set(['Ava', 'Ben', 'Cleo', 'Dev', 'Eli', 'Narrator'])

export default function Face({ name, className = 'avatar', fallback = '💀' }) {
  const [broken, setBroken] = useState(false)
  if (PORTRAITS.has(name) && !broken) {
    return <img className={`${className} portrait`} src={`/avatars/${name.toLowerCase()}.webp`} alt={name}
      draggable="false" onError={() => setBroken(true)} />
  }
  return <span className={className}>{EMOJI[name] ?? fallback}</span>
}
