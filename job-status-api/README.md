# Process Status Service

This is an isolated GCP Cloud Run function that retrieves logs for a specific process ID from Firestore and returns the status and important information about the process.

## Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [API Documentation](#api-documentation)
- [Error Handling](#error-handling)
- [Deployment](#deployment)
- [Local Development](#local-development)
- [Monitoring and Logging](#monitoring-and-logging)

## Features
- Get detailed status information for a specific process ID
- List all processes with filtering options
- Health check endpoint
- Firestore integration for data persistence
- Error tracking and detailed logging

## Prerequisites
- Python 3.8 or higher
- Google Cloud SDK
- Firebase Admin SDK credentials
- Access to Google Cloud Platform project
- Docker (for containerization)

## Environment Setup

1. Set up Google Cloud credentials:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account-key.json"
```

2. Configure environment variables:
```bash
export PROJECT_ID="your-gcp-project-id"
export FIRESTORE_COLLECTION="webServiceLogs"
```

## API Documentation

### Get Process Status
```
GET /status/{process_id}
```

Returns detailed information about a specific process.

**Query Parameters:**
- `include_products` (boolean): Include detailed product information (default: false)

**Success Response (200):**
```json
{
  "process_id": "abc123",
  "status": "completed",
  "start_time": "2023-06-01T12:00:00Z",
  "end_time": "2023-06-01T12:05:30Z",
  "last_updated": "2023-06-01T12:05:30Z",
  "csv_url": "https://example.com/products.csv",
  "products_processed": 10,
  "products_failed": 2
}
```

**Error Responses:**
- `404 Not Found`: Process ID doesn't exist
```json
{
  "status": "error",
  "message": "Process with ID abc123 not found"
}
```
- `500 Internal Server Error`: Server-side error
```json
{
  "status": "error",
  "message": "Error retrieving process status",
  "details": "<error stack trace>"
}
```

### List Processes
```
GET /status
```

Returns a list of processes with their basic status information.

**Query Parameters:**
- `limit` (integer): Maximum number of processes to return (default: 10)
- `status` (string): Filter by status (e.g., "completed", "failed", "processing")
- `start_date` (ISO date): Filter by start date
- `end_date` (ISO date): Filter by end date

**Success Response (200):**
```json
{
  "processes": [
    {
      "process_id": "abc123",
      "status": "completed",
      "start_time": "2023-06-01T12:00:00Z",
      "end_time": "2023-06-01T12:05:30Z",
      "last_updated": "2023-06-01T12:05:30Z",
      "products_processed": 10,
      "products_failed": 2
    }
  ],
  "count": 1
}
```

### Health Check
```
GET /
```

Returns a simple health check response.

**Success Response (200):**
```json
{
  "status": "healthy",
  "service": "process-status-service",
  "timestamp": "2023-06-01T12:00:00Z"
}
```

## Error Handling
The service implements comprehensive error handling:
- Input validation for all API parameters
- Detailed error messages and stack traces in development
- Sanitized error responses in production
- Automatic retry for transient Firestore errors

## Deployment

### Prerequisites
1. Enable required GCP APIs:
```bash
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

2. Configure Docker:
```bash
gcloud auth configure-docker
```

### Deployment Steps
1. Build the Docker image:
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/process-status-service
```

2. Deploy to Cloud Run:
```bash
gcloud run deploy process-status-service \
  --image gcr.io/PROJECT_ID/process-status-service \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "PROJECT_ID=your-gcp-project-id,FIRESTORE_COLLECTION=webServiceLogs"
```

3. Access the service at the URL provided in the deployment output.

## Local Development

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application with Gunicorn:
```bash
gunicorn main:app --bind 0.0.0.0:8080 --workers 4 --access-logfile -
```

Or for development with auto-reload:
```bash
python main.py
```

4. Access the service at http://localhost:8080

## Monitoring and Logging

### Cloud Run Monitoring
- View service metrics in Cloud Console
- Monitor request latency and error rates
- Set up alerts for critical metrics

### Logging
- All requests are logged to Cloud Logging
- Structured logging format for better analysis
- Error tracking with stack traces
- Custom log levels for different environments