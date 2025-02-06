FROM python:3.11.10-slim

# Install necessary system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt requirements-exact.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install -r requirements.txt \
    && pip install -r requirements-exact.txt

# Install your repository as a package
COPY . /app
WORKDIR /app
RUN pip install .

# Set the entrypoint (optional)
CMD ["bash"]