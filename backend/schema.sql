-- Tiger Data (TimescaleDB) setup. The app creates these tables itself on plain Postgres or SQLite;
-- run this once on Tiger Data first so `utterances` becomes a hypertable.
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    player TEXT,
    created_at DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS utterances (
    id SERIAL,
    session_id TEXT NOT NULL,
    t DOUBLE PRECISION NOT NULL,   -- unix seconds
    game INTEGER,
    day INTEGER,
    kind TEXT,                     -- calibration | answer
    question TEXT,
    transcript TEXT,
    role TEXT,                     -- mafia | villager
    raw JSON,                      -- face + voice measurements from the browser
    features JSON,                 -- z-scores vs the player's baseline + word cues
    is_lie BOOLEAN,
    p_lie DOUBLE PRECISION,        -- the model's read before it learned the label
    p_blind DOUBLE PRECISION,
    cues JSON
);
SELECT create_hypertable('utterances', by_range('t', 86400), if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS utterances_session_t ON utterances (session_id, t DESC);

-- Example: how the lie read evolved per minute across all players.
-- SELECT time_bucket(60, t) AS minute, avg(p_lie) FILTER (WHERE is_lie) AS lies, avg(p_lie) FILTER (WHERE NOT is_lie) AS truths
-- FROM utterances WHERE kind = 'answer' GROUP BY minute ORDER BY minute;
