# Use an official Python runtime as a parent image
FROM python:3.12-slim-bullseye

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    ffmpeg \
    espeak-ng libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

# Create templates directory
RUN mkdir -p /app/templates

# Copy template files if they exist
COPY templates /app/templates/

# Copy the current directory contents into the container at /app
COPY . /app

# Install project dependencies
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install -e .
RUN pip install gunicorn flask

# Set a default ACCESS_TOKEN (should be overridden in production)
ENV ACCESS_TOKEN="IV02ArtNTl2E-gTXNaEowxwsN2YqZ1E05SIO6BMqQxs"

# Add after pip installations and before CMD
RUN python -c "from TTS.utils.manage import ModelManager; ModelManager().download_model('tts_models/en/ljspeech/vits')"

# Add healthcheck with more generous parameters
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose the port the app runs on
EXPOSE 8000

# Use gunicorn with optimized settings for TTS workloads
CMD ["gunicorn", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "1", \
     "--threads", "4", \
     "--timeout", "300", \
     "--keepalive", "2", \
     "--max-requests", "1000", \
     "--max-requests-jitter", "50", \
     "--worker-class", "gthread", \
     "app:app"]
