import sqlite3


def main() -> None:
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE users
        SET minutes_quota = 30
        WHERE plan = 'free' AND minutes_quota = 10
        """
    )

    affected = cursor.rowcount
    conn.commit()

    print(f"\n{affected} Free-User auf 30 Minuten upgegraded\n")

    conn.close()


if __name__ == "__main__":
    main()
