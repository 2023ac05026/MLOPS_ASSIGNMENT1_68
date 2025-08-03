-- app/schema.sql
DROP TABLE IF EXISTS predictions;
DROP TABLE IF EXISTS metrics;

CREATE TABLE predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
  request_type TEXT NOT NULL,
  request_body TEXT NOT NULL,
  prediction_output TEXT NOT NULL,
  status TEXT NOT NULL
);

CREATE TABLE metrics (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
  cpu_percent REAL NOT NULL,
  memory_percent REAL NOT NULL,
  disk_percent REAL NOT NULL,
  avg_response_time_ms REAL NOT NULL
);