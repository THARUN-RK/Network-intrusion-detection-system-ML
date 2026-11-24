# Multi-stage Dockerfile for NIDS Backend and Streamlit Dashboard
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

# Expose FastAPI (8000) and Streamlit (8501) ports
EXPOSE 8000 8501

# Default command: run FastAPI backend
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
