web: gunicorn app:app --bind 0.0.0.0:$PORT
worker: rq worker -u $REDIS_URL default
