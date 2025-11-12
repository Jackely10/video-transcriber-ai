# asgi.py  (im Ordner, den Railway als Root Directory nutzt)
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/")
def index():
    return JSONResponse({"status": "ok", "service": "video-transcriber web"})
