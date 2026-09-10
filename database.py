import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent / "database" / "reqguard.db"


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                requirement_text TEXT NOT NULL,
                clarity REAL,
                completeness REAL,
                testability REAL,
                specificity REAL,
                ambiguity REAL,
                overall REAL,
                risk TEXT,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()


def save_analysis(text: str, result):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO analyses (
                requirement_text, clarity, completeness, testability,
                specificity, ambiguity, overall, risk, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            text, result.clarity, result.completeness, result.testability,
            result.specificity, result.ambiguity, result.overall,
            result.risk, datetime.now().isoformat(timespec="seconds")
        ))
        conn.commit()


def load_recent(limit: int = 100):
    with get_connection() as conn:
        cur = conn.execute("""
            SELECT id, requirement_text, clarity, completeness, testability,
                   specificity, ambiguity, overall, risk, created_at
            FROM analyses ORDER BY id DESC LIMIT ?
        """, (limit,))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
