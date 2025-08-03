-- app/schema.sql

-- This table stores logs for prediction requests.
-- 'IF NOT EXISTS' prevents an error if the table already exists.
CREATE TABLE IF NOT EXISTS predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
  request_type TEXT NOT NULL,
  request_body TEXT NOT NULL,
  prediction_output TEXT NOT NULL,
  status TEXT NOT NULL
);

-- This table can store historical metrics data.
-- 'IF NOT EXISTS' prevents an error if the table already exists.
CREATE TABLE IF NOT EXISTS metrics (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
  cpu_percent REAL NOT NULL,
  memory_percent REAL NOT NULL,
  disk_percent REAL NOT NULL,
  avg_response_time_ms REAL NOT NULL
);