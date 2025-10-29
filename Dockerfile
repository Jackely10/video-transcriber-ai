FROM python:3.11-slim

# Vor dem pip install
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
```

**Alternative Lösungen:**

1. **Verwenden Sie vorgefertigte Wheels**: Ändern Sie in `requirements.txt`:
```
   av>=11.0.0
