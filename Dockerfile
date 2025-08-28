# Use Python 3.11 slim image
FROM python:3.11-slim-bullseye

# Set working directory
WORKDIR /app

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install minimal system dependencies (no build tools needed)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies (skip TA-Lib, install everything else)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir yfinance==0.2.28 pandas==2.1.4 numpy==1.26.2 ta==0.10.2 requests==2.31.0 Flask==2.3.3 certifi==2023.11.17 charset-normalizer==3.3.2 idna==3.6 urllib3==2.1.0 python-dateutil==2.8.2 pytz==2023.3.post1 six==1.16.0 scipy==1.11.4

# Copy application code
COPY . .

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Expose port
EXPOSE 8080

# Default command
CMD ["python", "-u", "main.py"]
