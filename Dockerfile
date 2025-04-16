FROM python:3.12-slim

# Install FFmpeg
RUN apt-get update && apt-get install -y ffmpeg

# Upgrade pip + install your dependencies
COPY requirements.txt requirements.txt
COPY requirements-exact.txt requirements-exact.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Install your package
COPY . /app
WORKDIR /app
RUN pip install .

# Default command (won't be used in GitHub Actions but helps testing)
CMD ["python3"]