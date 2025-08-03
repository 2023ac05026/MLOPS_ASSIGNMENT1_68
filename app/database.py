# app/database.py
import sqlite3
import json
import os
from flask import g, current_app

DATABASE_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
DATABASE_PATH = os.path.join(DATABASE_FOLDER, 'logs.db')

def get_db():
    """
    Opens a new database connection if there is none yet for the current
    application context.
    """
    if 'db' not in g:
        os.makedirs(DATABASE_FOLDER, exist_ok=True)
        g.db = sqlite3.connect(DATABASE_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
        g.db.row_factory = sqlite3.Row
    return g.db

def close_db(e=None):
    """Closes the database again at the end of the request."""
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    """Initializes the database from the schema.sql file."""
    db = get_db()
    with current_app.open_resource('schema.sql') as f:
        db.executescript(f.read().decode('utf8'))

def log_prediction(request_type, request_body, prediction_output, status='SUCCESS'):
    """Logs a prediction request and its outcome to the SQLite database."""
    try:
        db = get_db()
        db.execute(
            'INSERT INTO predictions (request_type, request_body, prediction_output, status) VALUES (?, ?, ?, ?)',
            (request_type, json.dumps(request_body), json.dumps(prediction_output), status)
        )
        db.commit()
    except Exception as e:
        # Use the standard logger to log DB errors
        current_app.logger.error(f"Failed to log prediction to database: {e}")

def query_logs():
    """Queries all logs from the database, ordered by the most recent."""
    db = get_db()
    logs = db.execute(
        'SELECT id, timestamp, request_type, request_body, prediction_output, status '
        'FROM predictions ORDER BY timestamp DESC'
    ).fetchall()
    return [dict(log) for log in logs]

def init_app(app):
    """Register database functions with the Flask app."""
    app.teardown_appcontext(close_db)