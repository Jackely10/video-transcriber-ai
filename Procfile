web: uvicorn asgi:app --host 0.0.0.0 --port ${PORT:-8080}
worker: rq worker -u $REDIS_URL default
