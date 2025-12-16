# Use Python 3.11 slim image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code (from service directory context)
COPY . ./services/virtual_staging_api/

# Create __init__.py for services package
RUN echo "" > ./services/__init__.py

# Expose port
EXPOSE 8080

# Run the application
CMD ["uvicorn", "services.virtual_staging_api.main:app", "--host", "0.0.0.0", "--port", "8080"]
