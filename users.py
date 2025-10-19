"""User & Credits Management."""
from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

DB_PATH = Path("users.db")


def init_database() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(DB_PATH)) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                plan TEXT DEFAULT 'free',
                minutes_used REAL DEFAULT 0,
                minutes_quota REAL DEFAULT 30,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_reset_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS usage_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                job_id TEXT,
                video_duration REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS ip_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ip_address TEXT NOT NULL,
                email TEXT NOT NULL,
                first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                job_count INTEGER DEFAULT 0,
                UNIQUE(ip_address, email)
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ip_address ON ip_usage(ip_address)
            """
        )
        conn.commit()


class UserManager:
    """High level helper around the user credits database."""

    @staticmethod
    def get_or_create_user(email: str) -> Optional[Dict[str, object]]:
        with closing(sqlite3.connect(DB_PATH)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
            user = cursor.fetchone()
            if not user:
                cursor.execute("INSERT INTO users (email) VALUES (?)", (email,))
                conn.commit()
                cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
                user = cursor.fetchone()
            return dict(user) if user else None

    @staticmethod
    def _ensure_monthly_reset(minutes_used: float, last_reset: Optional[str], email: str) -> float:
        if not last_reset:
            UserManager.reset_monthly_quota(email)
            return 0.0
        try:
            last_reset_date = datetime.fromisoformat(last_reset)
        except ValueError:
            # If parsing fails, reset to keep system consistent.
            UserManager.reset_monthly_quota(email)
            return 0.0
        if datetime.now() - last_reset_date > timedelta(days=30):
            UserManager.reset_monthly_quota(email)
            return 0.0
        return minutes_used

    @staticmethod
    def check_quota(email: str, required_minutes: float) -> bool:
        with closing(sqlite3.connect(DB_PATH)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT minutes_used, minutes_quota, last_reset_at FROM users WHERE email = ?",
                (email,),
            )
            result = cursor.fetchone()
        if not result:
            return False

        minutes_used, minutes_quota, last_reset = result
        minutes_used = UserManager._ensure_monthly_reset(minutes_used, last_reset, email)
        return (minutes_quota - minutes_used) >= required_minutes

    @staticmethod
    def deduct_minutes(email: str, minutes: float, job_id: str) -> None:
        with closing(sqlite3.connect(DB_PATH)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE users SET minutes_used = minutes_used + ? WHERE email = ?",
                (minutes, email),
            )
            cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
            row = cursor.fetchone()
            user_id = row[0] if row else None
            if user_id is not None:
                cursor.execute(
                    "INSERT INTO usage_history (user_id, job_id, video_duration) VALUES (?, ?, ?)",
                    (user_id, job_id, minutes),
                )
            conn.commit()

    @staticmethod
    def reset_monthly_quota(email: str) -> None:
        with closing(sqlite3.connect(DB_PATH)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE users SET minutes_used = 0, last_reset_at = CURRENT_TIMESTAMP WHERE email = ?",
                (email,),
            )
            conn.commit()

    @staticmethod
    def upgrade_plan(email: str, plan: str) -> None:
        quotas = {"free": 30, "starter": 100, "pro": 500, "business": 2000}
        quota = quotas.get(plan, quotas["free"])
        with closing(sqlite3.connect(DB_PATH)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE users SET plan = ?, minutes_quota = ? WHERE email = ?",
                (plan, quota, email),
            )
            conn.commit()

    @staticmethod
    def get_user_stats(email: str) -> Optional[Dict[str, object]]:
        with closing(sqlite3.connect(DB_PATH)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT plan, minutes_used, minutes_quota FROM users WHERE email = ?",
                (email,),
            )
            result = cursor.fetchone()
            return dict(result) if result else None

    @staticmethod
    def check_ip_abuse(email: str, ip_address: str) -> bool:
        with closing(sqlite3.connect(DB_PATH)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(DISTINCT email) FROM ip_usage WHERE ip_address = ?",
                (ip_address,),
            )
            row = cursor.fetchone()
            count = row[0] if row else 0
            if count >= 3:
                cursor.execute(
                    "SELECT 1 FROM ip_usage WHERE ip_address = ? AND email = ?",
                    (ip_address, email),
                )
                exists = cursor.fetchone()
                return exists is not None
            return True

    @staticmethod
    def register_ip_usage(email: str, ip_address: str) -> None:
        with closing(sqlite3.connect(DB_PATH)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO ip_usage (ip_address, email, job_count, last_seen)
                VALUES (?, ?, 1, CURRENT_TIMESTAMP)
                ON CONFLICT(ip_address, email) DO UPDATE
                SET job_count = job_count + 1,
                    last_seen = CURRENT_TIMESTAMP
                """,
                (ip_address, email),
            )
            conn.commit()

    @staticmethod
    def get_ip_stats(ip_address: str) -> Dict[str, object]:
        with closing(sqlite3.connect(DB_PATH)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    COUNT(DISTINCT email) AS unique_emails,
                    COALESCE(SUM(job_count), 0) AS total_jobs,
                    MAX(last_seen) AS last_activity
                FROM ip_usage
                WHERE ip_address = ?
                """,
                (ip_address,),
            )
            result = cursor.fetchone()
            if not result:
                return {"unique_emails": 0, "total_jobs": 0, "last_activity": None}
            payload = dict(result)
            if payload.get("unique_emails") is None:
                payload["unique_emails"] = 0
            if payload.get("total_jobs") is None:
                payload["total_jobs"] = 0
            return payload

    @staticmethod
    def check_rate_limit(email: str) -> bool:
        with closing(sqlite3.connect(DB_PATH)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT plan FROM users WHERE email = ?", (email,))
            row = cursor.fetchone()
            if not row:
                return True
            plan = (row[0] or "").lower()
            if plan in {"starter", "pro", "business"}:
                return True
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM usage_history
                WHERE user_id = (SELECT id FROM users WHERE email = ?)
                  AND created_at > datetime('now', '-1 hour')
                """,
                (email,),
            )
            result = cursor.fetchone()
            count = result[0] if result else 0
            return count < 3

    @staticmethod
    def get_rate_limit_info(email: str) -> Dict[str, object]:
        with closing(sqlite3.connect(DB_PATH)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    u.plan,
                    COUNT(uh.id) AS jobs_last_hour
                FROM users u
                LEFT JOIN usage_history uh
                    ON u.id = uh.user_id
                    AND uh.created_at > datetime('now', '-1 hour')
                WHERE u.email = ?
                GROUP BY u.id
                """,
                (email,),
            )
            result = cursor.fetchone()
        if not result:
            return {"plan": "free", "jobs_last_hour": 0, "limit": 3, "remaining": 3}
        plan, jobs_last_hour = result
        plan = (plan or "free").lower()
        jobs_last_hour = int(jobs_last_hour or 0)
        if plan in {"starter", "pro", "business"}:
            return {
                "plan": plan,
                "jobs_last_hour": jobs_last_hour,
                "limit": None,
                "remaining": None,
            }
        limit = 3
        remaining = max(0, limit - jobs_last_hour)
        return {
            "plan": plan,
            "jobs_last_hour": jobs_last_hour,
            "limit": limit,
            "remaining": remaining,
        }


# Ensure database exists on import.
init_database()
