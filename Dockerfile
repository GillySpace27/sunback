FROM python:3.12-slim

# Install FFmpeg
RUN apt-get update && apt-get install -y ffmpeg

# Upgrade pip + install your dependencies
COPY requirements.txt requirements.txt
COPY requirements-exact.txt requirements-exact.txt
COPY requirements-server.txt requirements-server.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -r requirements-server.txt

# Set working directory
WORKDIR /app

# Default command (won't be used in GitHub Actions but helps testing)
CMD ["python3"]

LABEL org.opencontainers.image.source=https://github.com/GillySpace27/sunback