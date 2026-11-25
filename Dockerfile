# Multi-stage Dockerfile for NIDS Backend and Web Console
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies (for building packages and network utils)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tcpdump \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Default command: run FastAPI backend & web console
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
