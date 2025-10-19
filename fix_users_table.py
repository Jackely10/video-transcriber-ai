import sqlite3
con = sqlite3.connect("users.db")
cur = con.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS users (
  email TEXT PRIMARY KEY,
  plan TEXT NOT NULL DEFAULT 'starter',
  minutes_quota INTEGER NOT NULL DEFAULT 100,
  minutes_used INTEGER NOT NULL DEFAULT 0,
  created_at TEXT DEFAULT (datetime('now')),
  last_reset_at TEXT DEFAULT (datetime('now'))
);
""")
cur.execute("""
INSERT INTO users (email, plan, minutes_quota, minutes_used)
VALUES ('zouhir.bensebban@gmail.com','starter',100,0)
ON CONFLICT(email) DO UPDATE SET
  plan=excluded.plan,
  minutes_quota=excluded.minutes_quota,
  minutes_used=excluded.minutes_used;
""")
con.commit()
con.close()
print("users-Tabelle erstellt/aktualisiert.")
