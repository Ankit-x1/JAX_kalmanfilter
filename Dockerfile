# Professional JAX Kalman Filter - Docker Deployment

FROM python:3.11-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/app
ENV JAX_PLATFORMS=cpu

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    git \
    i2c-tools \
    python3-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/models /app/data /app/test_results

# Set permissions
RUN chmod +x main.py

# Install as package
RUN pip install -e .

# Expose port for potential web interface
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from config.config_manager import config_manager; print('OK')" || exit 1

# Default command
CMD ["python", "main.py", "--mode", "deploy", "--duration", "60"]