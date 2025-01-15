FROM python:3.10-slim-bullseye

# Install system dependencies in a single RUN command to reduce layers
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    python3-dev \
    libsndfile1 \
    ffmpeg \
    espeak \
    build-essential \
    cmake \
    curl \
    pkg-config \
    git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install requirements in separate layers
COPY requirements.txt .

# Install PyTorch CPU version
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
RUN pip install --no-cache-dir -r requirements.txt

# Create directories before copying files
RUN mkdir -p Data/podcast_episodes Data/voice_Data

# Copy application files
COPY app/ app/
COPY Data/ Data/

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]