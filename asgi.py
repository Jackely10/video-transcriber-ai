"""ASGI entrypoint for running the Flask app with Uvicorn."""

from asgiref.wsgi import WsgiToAsgi

from server import app as flask_app

# Wrap the existing WSGI Flask app so it can be served by an ASGI server.
app = WsgiToAsgi(flask_app)

