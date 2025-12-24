FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY firebase_config.py .
COPY furniture_scraper.py .
COPY gunicorn_config.py .

EXPOSE 8080

# Default to furniture scraper
CMD ["gunicorn", "--config", "gunicorn_config.py", "furniture_scraper:app"]