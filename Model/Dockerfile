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

# Pin numpy<2 first — torch 2.x compiled modules crash with numpy>=2
RUN pip install --no-cache-dir "numpy>=1.24.4,<2"

# Install CPU-only torch (2.4.1 — minimum required by sentence-transformers/transformers)
RUN pip install --no-cache-dir \
    torch==2.4.1+cpu \
    torchaudio==2.4.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining deps (numpy already pinned, won't be upgraded)
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8001

CMD ["python", "scripts/serve.py", "--port", "8001"]
