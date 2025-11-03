# wsgi.py
# Versucht zuerst den Paketpfad, fällt sonst auf server.py im Root zurück.
try:
    from video_transcriber.server import app as app
except ModuleNotFoundError:
    from server import app as app  # falls server.py im Repo-Root liegt
