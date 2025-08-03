DROP TABLE IF EXISTS predictions;

CREATE TABLE predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
  request_type TEXT NOT NULL,
  request_body TEXT NOT NULL,
  prediction_output TEXT NOT NULL,
  status TEXT NOT NULL
);