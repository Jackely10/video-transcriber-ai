# wsgi.py — stabiler Einstieg für Gunicorn
# Versucht zuerst "video_transcriber.server:app", fällt sonst
# auf "server:app" oder "app:app" zurück.

try:
    from video_transcriber.server import app as app  # Paket/Unterordner
except Exception:
    try:
        from server import app as app  # Datei im Repo-Root
    except Exception:
        from app import app as app 
