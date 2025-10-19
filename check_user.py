import os, sqlite3, sys

DB = "users.db"
EMAIL = "zouhir.bensebban@gmail.com"

if not os.path.exists(DB):
    print(f"[!] Datei {DB} nicht gefunden in {os.getcwd()}")
    sys.exit(1)

con = sqlite3.connect(DB)
con.row_factory = sqlite3.Row
cur = con.cursor()

try:
    cur.execute("PRAGMA table_info(users)")
    cols = [row["name"] for row in cur.fetchall()]
    print("Spalten in 'users':", ", ".join(cols) if cols else "(Tabelle fehlt)")

    cur.execute(
        "SELECT email, plan, minutes_quota, minutes_used FROM users WHERE email = ?",
        (EMAIL,),
    )
    row = cur.fetchone()
    if row:
        quota = row["minutes_quota"] or 0
        used = row["minutes_used"] or 0
        print("\n=== USER INFO ===")
        print(f"Email:      {row['email']}")
        print(f"Plan:       {row['plan']}")
        print(f"Quota:      {quota} Minuten")
        print(f"Benutzt:    {used} Minuten")
        print(f"Verfügbar:  {quota - used} Minuten")
    else:
        print(f"User {EMAIL} nicht gefunden. Beispiel-Auszug:")
        cur.execute("SELECT email, plan, minutes_quota, minutes_used FROM users LIMIT 5")
        for r in cur.fetchall():
            print(dict(r))
except sqlite3.OperationalError as e:
    print(f"[!] SQLite-Fehler: {e}")
    print("Tipp: Wurde die Tabelle 'users' schon angelegt? App einmal starten oder Init-Script ausführen.")
finally:
    con.close()
