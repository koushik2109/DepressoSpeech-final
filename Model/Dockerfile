FROM python:3.11-slim

WORKDIR /app

# System deps: libgl (mediapipe/opencv), ffmpeg (audio), libgomp (torch)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only torch — latest from wheel index (no version pin)
# Same approach as start-all.sh: installs latest compatible cpu set
RUN pip install --no-cache-dir \
    torch torchaudio torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining deps (torch already satisfied above, won't re-download)
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8001

CMD ["python", "scripts/serve.py", "--port", "8001"]
