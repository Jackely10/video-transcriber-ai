FROM python:3.11

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavdevice-dev \
    libavutil-dev \
    libavfilter-dev \
    libswscale-dev \
    libswresample-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENV PORT=8080

EXPOSE 8080

CMD ["sh", "-c", "uvicorn asgi:app --host 0.0.0.0 --port ${PORT:-8080}"]
