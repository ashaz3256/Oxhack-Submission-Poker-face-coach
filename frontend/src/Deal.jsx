// The deal: six face-down cards. You say a number, your card flips to reveal your role, and the
// other five slide off to the AI players.
export default function Deal({ picked, youMafia, cards }) {
  return (
    <div className="deal-overlay">
      <div className="deal-title">{picked ? 'Your card' : 'Say a number from one to six'}</div>
      <div className="deal-row">
        {[1, 2, 3, 4, 5, 6].map((n, i) => {
          const mine = picked === n
          const owner = cards?.[i]
          const gone = picked && !mine
          return (
            <div key={n} className={`deal-card ${mine ? 'mine' : ''} ${gone ? 'gone' : ''}`}
              style={{ animationDelay: `${i * 90}ms`, transitionDelay: gone ? `${300 + i * 120}ms` : '0ms' }}>
              <div className={`deal-inner ${mine && youMafia !== null ? 'flipped' : ''}`}>
                <div className="deal-face deal-back">
                  <span className="deal-num">{n}</span>
                  <span className="deal-logo">TILT</span>
                </div>
                <div className={`deal-face deal-front ${youMafia ? 'mafia' : 'villager'}`}>
                  <span className="deal-icon">{youMafia ? '🎭' : '🏡'}</span>
                  <span className="deal-role">{youMafia ? 'MAFIA' : 'VILLAGER'}</span>
                  <span className="deal-sub">{youMafia ? 'Lie. Survive.' : 'Find the liar.'}</span>
                </div>
              </div>
              {gone && owner && <div className="deal-owner">→ {owner}</div>}
            </div>
          )
        })}
      </div>
    </div>
  )
}
