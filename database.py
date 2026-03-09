"""
database.py
-----------
SQLite database setup and management for the Solar Microgrid AI System.
Handles user table creation and initial seed data.
"""

import sqlite3
import hashlib
import os

DB_PATH = "solar_microgrid.db"


def get_connection():
    """Return a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def hash_password(password: str) -> str:
    """Return SHA-256 hash of a password string."""
    return hashlib.sha256(password.encode()).hexdigest()


def init_db():
    """
    Initialize the database.
    Creates the users table and seeds default admin + user accounts
    if they don't already exist.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Create users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT    NOT NULL UNIQUE,
            password TEXT    NOT NULL,
            role     TEXT    NOT NULL CHECK(role IN ('admin', 'user'))
        )
    """)

    # Create predictions log table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT    NOT NULL,
            timestamp     DATETIME DEFAULT CURRENT_TIMESTAMP,
            solar_irr     REAL,
            cloud_cover   REAL,
            battery_level REAL,
            total_load    REAL,
            prediction    TEXT
        )
    """)

    # Seed default accounts
    default_users = [
        ("admin",   hash_password("admin123"),  "admin"),
        ("operator", hash_password("user123"),  "user"),
    ]
    for username, pwd, role in default_users:
        cursor.execute(
            "INSERT OR IGNORE INTO users (username, password, role) VALUES (?, ?, ?)",
            (username, pwd, role),
        )

    conn.commit()
    conn.close()


# ── User CRUD ──────────────────────────────────────────────────────────────────

def get_all_users():
    """Return all users as a list of sqlite3.Row objects."""
    conn = get_connection()
    users = conn.execute("SELECT id, username, role FROM users").fetchall()
    conn.close()
    return users


def add_user(username: str, password: str, role: str) -> bool:
    """Insert a new user. Returns True on success, False if username exists."""
    try:
        conn = get_connection()
        conn.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            (username, hash_password(password), role),
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False


def delete_user(user_id: int):
    """Delete a user by ID."""
    conn = get_connection()
    conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()


def verify_user(username: str, password: str):
    """
    Check credentials.
    Returns the user row on success, or None on failure.
    """
    conn = get_connection()
    user = conn.execute(
        "SELECT * FROM users WHERE username = ? AND password = ?",
        (username, hash_password(password)),
    ).fetchone()
    conn.close()
    return user


# ── Prediction Logging ─────────────────────────────────────────────────────────

def log_prediction(username, solar_irr, cloud_cover, battery_level, total_load, prediction):
    """Persist a prediction event to the database."""
    conn = get_connection()
    conn.execute(
        """INSERT INTO prediction_logs
           (username, solar_irr, cloud_cover, battery_level, total_load, prediction)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (username, solar_irr, cloud_cover, battery_level, total_load, prediction),
    )
    conn.commit()
    conn.close()


def get_prediction_logs(limit: int = 50):
    """Return the most recent prediction log entries."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM prediction_logs ORDER BY timestamp DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return rows
