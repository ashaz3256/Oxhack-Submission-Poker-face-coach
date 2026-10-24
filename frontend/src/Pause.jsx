export default function Pause({ onResume }) {
  return (
    <div className="pause" onClick={onResume}>
      <div className="pause-box" onClick={(e) => e.stopPropagation()}>
        <div className="pause-title">Paused</div>
        <p className="hint">Everyone's frozen. Nothing you say now is recorded.</p>
        <button className="primary big" onClick={onResume}>▶ Resume</button>
        <p className="hint">Or press Esc</p>
      </div>
    </div>
  )
}
