FROM python:3.11-slim

# ffmpeg installieren
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python-Abhängigkeiten
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code kopieren
COPY . .

# Startbefehl für Web (in Railway pro Service überschreibbar)
CMD ["gunicorn","wsgi:app","--bind","0.0.0.0:${PORT}","--workers","1","--timeout","120"]
