-- Tiger Data (TimescaleDB) setup. The app creates these tables itself on plain Postgres or SQLite;
-- run this once on Tiger Data first so `vitals` becomes a hypertable.
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    player TEXT,
    opponent TEXT,
    vitals_source TEXT,
    created_at DOUBLE PRECISION,
    net_bb DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS vitals (
    session_id TEXT NOT NULL,
    t DOUBLE PRECISION NOT NULL,   -- unix seconds
    hr DOUBLE PRECISION,
    br DOUBLE PRECISION
);
SELECT create_hypertable('vitals', by_range('t', 3600), if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS vitals_session_t ON vitals (session_id, t DESC);

-- Example: heart rate per 5 seconds around one session's decisions.
-- SELECT time_bucket(5, t) AS bucket, avg(hr) FROM vitals WHERE session_id = '...' GROUP BY bucket ORDER BY bucket;
