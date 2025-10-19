# -*- coding: utf-8 -*-
"""Minimal Flask admin dashboard for monitoring user activity."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional

from flask import Flask, render_template_string

DB_PATH = "users.db"
PLAN_LABELS = {
    "free": "Free",
    "starter": "Starter",
    "pro": "Pro",
    "business": "Business",
}
PLAN_PRICES_EUR = {
    "free": 0.0,
    "starter": 9.0,
    "pro": 29.0,
    "business": 99.0,
}

app = Flask(__name__)

ADMIN_DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Admin Dashboard - Video Transcriber</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
            color: #1a202c;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        .subtitle {
            color: #666;
            margin-bottom: 30px;
        }
        .section {
            margin-top: 40px;
            margin-bottom: 32px;
        }
        .section-title {
            font-size: 24px;
            color: #333;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
            display: flex;
            flex-direction: column;
            gap: 6px;
        }
        .stat-label {
            font-size: 14px;
            opacity: 0.9;
            margin-bottom: 8px;
        }
        .stat-value {
            font-size: 36px;
            font-weight: 800;
            color: inherit;
        }
        .table-wrapper {
            background: #fff;
            border-radius: 18px;
            border: 1px solid rgba(160, 174, 192, 0.35);
            overflow: hidden;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.3);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            background: white;
            border-radius: 10px;
            overflow: hidden;
        }
        th, td {
            text-align: left;
            font-size: 14px;
            color: #2d3748;
        }
        td {
            padding: 12px 15px;
            border-bottom: 1px solid #f0f0f0;
        }
        th {
            background: #667eea;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }
        tr:nth-child(even) td {
            background: rgba(247, 250, 252, 0.8);
        }
        tr:hover td {
            background: #f9f9f9;
        }
        .badge {
            display: inline-flex;
            align-items: center;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            background: rgba(102, 126, 234, 0.12);
            color: #434190;
        }
        .badge-free,
        .badge.free { background: #e0e0e0; color: #666; }
        .badge-starter,
        .badge.starter { background: #d4edda; color: #155724; }
        .badge-pro,
        .badge.pro { background: #cce5ff; color: #004085; }
        .badge-business,
        .badge.business { background: #f8d7da; color: #721c24; }
        .empty {
            padding: 18px;
            text-align: center;
            color: #718096;
            font-style: italic;
        }
        .grid-two {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 16px;
        }
        .card {
            background: #fff;
            border-radius: 18px;
            border: 1px solid rgba(160, 174, 192, 0.35);
            padding: 20px;
            box-shadow: 0 10px 20px rgba(15, 30, 65, 0.15);
        }
        .card h3 {
            font-size: 18px;
            margin-bottom: 12px;
            color: #2d3748;
        }
        .trend-list li {
            list-style: none;
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid rgba(226, 232, 240, 0.6);
            font-size: 14px;
            color: #2d3748;
        }
        .trend-list li:last-child {
            border-bottom: none;
        }
        .trend-list span {
            display: inline-block;
            min-width: 110px;
        }
        .muted {
            color: #718096;
            font-size: 13px;
        }
        .progress-bar {
            width: 100%;
            height: 8px;
            background: #e0e0e0;
            border-radius: 4px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            background: #667eea;
            transition: width 0.3s;
        }
        .refresh-btn {
            position: fixed;
            bottom: 30px;
            right: 30px;
            padding: 15px 30px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 50px;
            cursor: pointer;
            font-weight: 600;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        }
        .refresh-btn:hover {
            transform: translateY(-2px);
        }
        @media (max-width: 768px) {
            .stats-grid {
                grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
            }
            th, td {
                font-size: 13px;
                padding: 10px 12px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎛 Admin Dashboard</h1>
        <p class="subtitle">Video Transcriber - User &amp; Revenue Übersicht</p>
        <p class="muted">Stand: {{ generated_at }}</p>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">👥 Gesamt User</div>
                <div class="stat-value">{{ stats.total_users }}</div>
                <div class="muted">{{ stats.active_24h }} aktiv in den letzten 24h</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">💰 Zahlende Kunden</div>
                <div class="stat-value">{{ stats.paid_users }}</div>
                <div class="muted">{{ stats.free_users }} im Free-Plan</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">📊 Gesamt Minuten</div>
                <div class="stat-value">{{ "{:.1f}".format(stats.total_minutes_used) }}</div>
                <div class="muted">{{ "{:.1f}".format(stats.minutes_last_7_days) }} Min in 7 Tagen</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">💵 MRR (geschätzt)</div>
                <div class="stat-value">{{ "{:.2f} €".format(stats.mrr) }}</div>
                <div class="muted">Ø {{ "{:.1f}".format(stats.avg_minutes_per_job) }} Min pro Job</div>
            </div>
        </div>

        <div class="section">
            <h2 class="section-title">📋 Alle User</h2>
            <div class="table-wrapper">
                <table>
                    <thead>
                        <tr>
                            <th>Email</th>
                            <th>Plan</th>
                            <th>Credits</th>
                            <th>Nutzung</th>
                            <th>Jobs (24h)</th>
                            <th>Registriert</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for user in users %}
                            <tr>
                                <td>{{ user.email }}</td>
                                <td>
                                    <span class="badge badge-{{ user.plan_slug }}">{{ user.plan_label }}</span>
                                </td>
                                <td>{{ user.minutes_remaining }} / {{ user.minutes_quota }} Min</td>
                                <td>
                                    <div class="progress-bar">
                                        <div class="progress-fill" style="width: {{ user.usage_percent }};"></div>
                                    </div>
                                    <div class="muted">{{ user.minutes_used }} Min genutzt</div>
                                </td>
                                <td>{{ user.jobs_24h }}</td>
                                <td>{{ user.created_at }}</td>
                            </tr>
                        {% else %}
                            <tr>
                                <td colspan="6" class="empty">Keine Nutzer gefunden.</td>
                            </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>

        <div class="section">
            <h2 class="section-title">📈 Plan-Verteilung</h2>
            {% if plan_breakdown %}
                <div class="table-wrapper">
                    <table>
                        <thead>
                            <tr>
                                <th>Plan</th>
                                <th>User</th>
                                <th>Anteil</th>
                                <th>MRR</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for item in plan_breakdown %}
                                <tr>
                                    <td><span class="badge badge-{{ item.slug }}">{{ item.plan }}</span></td>
                                    <td>{{ item.count }}</td>
                                    <td>{{ item.percentage }}%</td>
                                    <td>{{ "{:.2f} €".format(item.revenue) }}</td>
                                </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            {% else %}
                <div class="empty">Noch keine Nutzerstatistik verfügbar.</div>
            {% endif %}
        </div>

        <div class="section grid-two">
            <div class="card">
                <h3>⚠️ Niedrige Kontingente</h3>
                {% if near_quota %}
                    <div class="table-wrapper">
                        <table>
                            <thead>
                                <tr>
                                    <th>Email</th>
                                    <th>Plan</th>
                                    <th>Rest</th>
                                    <th>Zuletzt Reset</th>
                                </tr>
                            </thead>
                            <tbody>
                                {% for user in near_quota %}
                                    <tr>
                                        <td>{{ user.email }}</td>
                                        <td><span class="badge badge-{{ user.raw_plan }}">{{ user.plan }}</span></td>
                                        <td>{{ user.minutes_remaining }} Min</td>
                                        <td>{{ user.last_reset_at }}</td>
                                    </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                {% else %}
                    <div class="empty">Alle User haben ausreichend Minuten.</div>
                {% endif %}
            </div>
            <div class="card">
                <h3>🌐 IP-Aktivität</h3>
                {% if ip_alerts %}
                    <ul class="trend-list">
                        {% for ip in ip_alerts %}
                            <li>
                                <div>
                                    <strong>{{ ip.ip_address }}</strong>
                                    <div class="muted">{{ ip.email_count }} Accounts • {{ ip.total_jobs }} Jobs</div>
                                </div>
                                <span>{{ ip.last_seen }}</span>
                            </li>
                        {% endfor %}
                    </ul>
                {% else %}
                    <div class="empty">Keine IP-Auffälligkeiten vorhanden.</div>
                {% endif %}
            </div>
        </div>

        <div class="section">
            <h2 class="section-title">📈 Letzte Jobs (24h)</h2>
            {% if recent_jobs %}
                <div class="table-wrapper">
                    <table>
                        <thead>
                            <tr>
                                <th>Email</th>
                                <th>Job ID</th>
                                <th>Dauer (Min)</th>
                                <th>Zeitpunkt</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for job in recent_jobs %}
                                <tr>
                                    <td>{{ job.email }}</td>
                                    <td><code>{{ job.job_id_short }}</code></td>
                                    <td>{{ job.minutes_2f }}</td>
                                    <td>{{ job.created_at }}</td>
                                </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            {% else %}
                <div class="empty">Keine Jobs in den letzten 24 Stunden.</div>
            {% endif %}
        </div>

        <div class="section">
            <h2 class="section-title">📊 Nutzungsverlauf (7 Tage)</h2>
            <div class="card">
                {% if usage_trend %}
                    <ul class="trend-list">
                        {% for day in usage_trend %}
                            <li>
                                <div>
                                    <strong>{{ day.label }}</strong>
                                    <div class="muted">{{ day.jobs }} Jobs</div>
                                </div>
                                <span>{{ day.minutes }} Min</span>
                            </li>
                        {% endfor %}
                    </ul>
                {% else %}
                    <div class="empty">Keine Daten für den Zeitraum verfügbar.</div>
                {% endif %}
            </div>
        </div>
    </div>

    <button class="refresh-btn" onclick="location.reload()">Aktualisieren</button>
</body>
</html>
"""

def fetch_rows(query: str, params: Iterable[Any] = ()) -> List[Dict[str, Any]]:
    """Execute a SELECT query and return rows as dictionaries."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()
    except sqlite3.Error:
        return []
    return [dict(row) for row in rows]


def fetch_one(query: str, params: Iterable[Any] = ()) -> Optional[Dict[str, Any]]:
    rows = fetch_rows(query, params)
    return rows[0] if rows else None


def format_timestamp(value: Optional[str]) -> str:
    if not value:
        return "-"
    dt: Optional[datetime] = None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(value, fmt)
            break
        except ValueError:
            dt = None
    if not dt:
        try:
            dt = datetime.fromisoformat(value)
        except ValueError:
            return value
    return dt.strftime("%d.%m.%Y %H:%M")


def format_number(value: Optional[float], decimals: int = 1) -> str:
    if value is None:
        return "-"
    formatted = f"{float(value):.{decimals}f}"
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted or "0"


def sanitize_plan(plan: Optional[str]) -> str:
    key = (plan or "free").lower()
    return PLAN_LABELS.get(key, key.title() or "Free")


def get_stats() -> Dict[str, Any]:
    total_users = (fetch_one("SELECT COUNT(*) AS cnt FROM users") or {}).get("cnt", 0) or 0
    total_users = int(total_users)

    plan_counts = fetch_rows(
        """
        SELECT LOWER(COALESCE(plan, 'free')) AS plan, COUNT(*) AS cnt
        FROM users
        GROUP BY LOWER(COALESCE(plan, 'free'))
        """
    )
    paid_users = sum(int(row.get("cnt", 0) or 0) for row in plan_counts if (row.get("plan") or "").lower() in {"starter", "pro", "business"})
    free_users = max(total_users - paid_users, 0)

    new_users_last_7_days = (
        fetch_one(
            """
            SELECT COUNT(*) AS cnt
            FROM users
            WHERE datetime(created_at) >= datetime('now', '-7 days')
            """
        )
        or {}
    ).get("cnt", 0) or 0
    new_users_last_7_days = int(new_users_last_7_days)

    active_24h = (
        fetch_one(
            """
            SELECT COUNT(DISTINCT user_id) AS cnt
            FROM usage_history
            WHERE datetime(created_at) >= datetime('now', '-1 day')
            """
        )
        or {}
    ).get("cnt", 0) or 0
    active_24h = int(active_24h)

    jobs_last_7_days_row = fetch_one(
        """
        SELECT COUNT(*) AS jobs, COALESCE(SUM(video_duration), 0) AS minutes
        FROM usage_history
        WHERE datetime(created_at) >= datetime('now', '-7 days')
        """
    ) or {}
    jobs_last_7_days = int(jobs_last_7_days_row.get("jobs", 0) or 0)
    minutes_last_7_days = float(jobs_last_7_days_row.get("minutes", 0.0) or 0.0)

    total_minutes_row = fetch_one(
        """
        SELECT COALESCE(SUM(video_duration), 0) AS minutes
        FROM usage_history
        """
    ) or {}
    total_minutes_used = float(total_minutes_row.get("minutes", 0.0) or 0.0)

    avg_minutes_per_job = minutes_last_7_days / jobs_last_7_days if jobs_last_7_days else 0.0

    paying_conversion = (paid_users / new_users_last_7_days * 100) if new_users_last_7_days else 0.0

    mrr = 0.0
    for row in plan_counts:
        plan_key = (row.get("plan") or "free").lower()
        count = int(row.get("cnt", 0) or 0)
        mrr += PLAN_PRICES_EUR.get(plan_key, 0.0) * count

    return {
        "total_users": total_users,
        "paid_users": paid_users,
        "free_users": free_users,
        "new_users_last_7_days": new_users_last_7_days,
        "active_24h": active_24h,
        "jobs_last_7_days": jobs_last_7_days,
        "minutes_last_7_days": minutes_last_7_days,
        "total_minutes_used": total_minutes_used,
        "avg_minutes_per_job": avg_minutes_per_job,
        "paying_conversion": paying_conversion,
        "mrr": mrr,
    }


def get_plan_breakdown(total_users: int) -> List[Dict[str, Any]]:
    rows = fetch_rows(
        """
        SELECT LOWER(COALESCE(plan, 'free')) AS plan, COUNT(*) AS cnt
        FROM users
        GROUP BY LOWER(COALESCE(plan, 'free'))
        ORDER BY cnt DESC
        """
    )
    breakdown: List[Dict[str, Any]] = []
    for row in rows:
        plan_key = (row.get("plan") or "free").lower()
        count = int(row.get("cnt", 0) or 0)
        percentage = (count / total_users * 100) if total_users else 0.0
        revenue = PLAN_PRICES_EUR.get(plan_key, 0.0) * count
        breakdown.append(
            {
                "plan": sanitize_plan(plan_key),
                "slug": plan_key,
                "count": count,
                "percentage": format_number(percentage, decimals=1),
                "revenue": revenue,
            }
        )
    return breakdown


def get_users_overview(limit: int = 50) -> List[Dict[str, Any]]:
    rows = fetch_rows(
        """
        SELECT
            u.id,
            u.email,
            LOWER(COALESCE(u.plan, 'free')) AS plan,
            COALESCE(u.minutes_used, 0) AS minutes_used,
            COALESCE(u.minutes_quota, 0) AS minutes_quota,
            u.created_at,
            (
                SELECT COUNT(*)
                FROM usage_history uh
                WHERE uh.user_id = u.id
                  AND datetime(uh.created_at) >= datetime('now', '-1 day')
            ) AS jobs_24h
        FROM users u
        ORDER BY datetime(u.created_at) DESC
        LIMIT ?
        """,
        (limit,),
    )
    overview: List[Dict[str, Any]] = []
    for row in rows:
        plan_key = (row.get("plan") or "free").lower()
        minutes_used = float(row.get("minutes_used", 0.0) or 0.0)
        minutes_quota = float(row.get("minutes_quota", 0.0) or 0.0)
        remaining = max(minutes_quota - minutes_used, 0.0)
        usage_percent = (minutes_used / minutes_quota * 100) if minutes_quota else 0.0
        usage_percent = max(0.0, min(usage_percent, 100.0))
        overview.append(
            {
                "email": row.get("email", "—"),
                "plan_label": sanitize_plan(plan_key),
                "plan_slug": plan_key,
                "raw_plan": plan_key,
                "minutes_remaining": format_number(remaining),
                "minutes_quota": format_number(minutes_quota),
                "minutes_used": format_number(minutes_used),
                "usage_percent": f"{usage_percent:.0f}%",
                "jobs_24h": int(row.get("jobs_24h", 0) or 0),
                "created_at": format_timestamp(row.get("created_at")),
            }
        )
    return overview


def get_near_quota(limit: int = 5, threshold_minutes: float = 10.0) -> List[Dict[str, Any]]:
    fetch_limit = max(limit * 3, limit)
    rows = fetch_rows(
        """
        SELECT
            email,
            LOWER(COALESCE(plan, 'free')) AS plan,
            COALESCE(minutes_used, 0) AS minutes_used,
            COALESCE(minutes_quota, 0) AS minutes_quota,
            last_reset_at
        FROM users
        WHERE COALESCE(minutes_quota, 0) > 0
        ORDER BY (COALESCE(minutes_quota, 0) - COALESCE(minutes_used, 0)) ASC
        LIMIT ?
        """,
        (fetch_limit,),
    )
    alert_list: List[Dict[str, Any]] = []
    for row in rows:
        minutes_used = float(row.get("minutes_used", 0.0) or 0.0)
        minutes_quota = float(row.get("minutes_quota", 0.0) or 0.0)
        remaining = max(minutes_quota - minutes_used, 0.0)
        if remaining > threshold_minutes:
            continue
        plan_key = (row.get("plan") or "free").lower()
        alert_list.append(
            {
                "email": row.get("email", "—"),
                "plan": sanitize_plan(plan_key),
                "raw_plan": plan_key,
                "minutes_remaining": format_number(remaining),
                "last_reset_at": format_timestamp(row.get("last_reset_at")),
            }
        )
        if len(alert_list) >= limit:
            break
    return alert_list


def get_recent_jobs(limit: int = 25) -> List[Dict[str, Any]]:
    rows = fetch_rows(
        """
        SELECT
            COALESCE(uh.job_id, '-') AS job_id,
            COALESCE(u.email, 'Unbekannt') AS email,
            COALESCE(uh.video_duration, 0) AS minutes,
            uh.created_at
        FROM usage_history uh
        LEFT JOIN users u ON u.id = uh.user_id
        WHERE datetime(uh.created_at) >= datetime('now', '-1 day')
        ORDER BY datetime(uh.created_at) DESC
        LIMIT ?
        """,
        (limit,),
    )
    jobs: List[Dict[str, Any]] = []
    for row in rows:
        job_id = row.get("job_id", "-") or "-"
        email = row.get("email", "Unbekannt")
        minutes_raw = float(row.get("minutes", 0.0) or 0.0)
        created_at = format_timestamp(row.get("created_at"))
        short_id = job_id[:8] + "..." if len(job_id) > 8 else job_id
        jobs.append(
            {
                "job_id": job_id,
                "job_id_short": short_id,
                "email": email,
                "minutes_2f": f"{minutes_raw:.2f}",
                "created_at": created_at,
            }
        )
    return jobs


def get_ip_alerts(limit: int = 6) -> List[Dict[str, Any]]:
    rows = fetch_rows(
        """
        SELECT
            ip_address,
            COUNT(*) AS email_count,
            COALESCE(SUM(job_count), 0) AS total_jobs,
            MAX(last_seen) AS last_seen
        FROM ip_usage
        GROUP BY ip_address
        ORDER BY total_jobs DESC, email_count DESC
        LIMIT ?
        """,
        (limit,),
    )
    alerts: List[Dict[str, Any]] = []
    for row in rows:
        alerts.append(
            {
                "ip_address": row.get("ip_address", "-"),
                "email_count": int(row.get("email_count", 0) or 0),
                "total_jobs": int(row.get("total_jobs", 0) or 0),
                "last_seen": format_timestamp(row.get("last_seen")),
            }
        )
    return alerts


def get_usage_trend(days: int = 7) -> List[Dict[str, Any]]:
    start_date = datetime.utcnow().date() - timedelta(days=days - 1)
    rows = fetch_rows(
        """
        SELECT DATE(created_at) AS day, COUNT(*) AS jobs, COALESCE(SUM(video_duration), 0) AS minutes
        FROM usage_history
        WHERE DATE(created_at) >= DATE('now', ?)
        GROUP BY DATE(created_at)
        ORDER BY day ASC
        """,
        (f"-{days - 1} days",),
    )
    data_by_day = {row.get("day"): row for row in rows if row.get("day")}
    trend: List[Dict[str, Any]] = []
    for i in range(days):
        current_date = start_date + timedelta(days=i)
        key = current_date.isoformat()
        row = data_by_day.get(key, {"jobs": 0, "minutes": 0})
        trend.append(
            {
                "label": current_date.strftime("%a, %d.%m."),
                "jobs": int(row.get("jobs", 0) or 0),
                "minutes": format_number(row.get("minutes", 0.0)),
            }
        )
    return trend


@app.route("/admin")
def admin_dashboard():
    stats = get_stats()
    plan_breakdown = get_plan_breakdown(stats.get("total_users", 0))
    users = get_users_overview()
    near_quota = get_near_quota()
    ip_alerts = get_ip_alerts()
    recent_jobs = get_recent_jobs()
    usage_trend = get_usage_trend()
    generated_at = datetime.now().strftime("%d.%m.%Y %H:%M")
    return render_template_string(
        ADMIN_DASHBOARD_HTML,
        stats=stats,
        users=users,
        plan_breakdown=plan_breakdown,
        near_quota=near_quota,
        ip_alerts=ip_alerts,
        recent_jobs=recent_jobs,
        usage_trend=usage_trend,
        generated_at=generated_at,
    )


if __name__ == "__main__":
    print("🎛  Admin Dashboard läuft auf: http://localhost:5001/admin")
    app.run(port=5001, debug=True)
