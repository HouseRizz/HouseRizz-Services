#!/bin/bash
set -e

PROJECT_ID="houserizz-481012"
REGION="us-central1"
SERVICE_NAME="furniture-scraper"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "=========================================="
echo "Deploying Furniture Scraper to Cloud Run"
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Service: ${SERVICE_NAME}"
echo "=========================================="

# Enable required APIs
echo "Enabling required APIs..."
gcloud services enable run.googleapis.com --project ${PROJECT_ID}
gcloud services enable aiplatform.googleapis.com --project ${PROJECT_ID}
gcloud services enable firestore.googleapis.com --project ${PROJECT_ID}
gcloud services enable storage.googleapis.com --project ${PROJECT_ID}

# Build locally with linux/amd64 platform (required for Cloud Run on Mac M1/ARM)
echo "Building Docker image for linux/amd64..."
docker build --platform linux/amd64 -t ${IMAGE} .

echo "Pushing to GCR..."
docker push ${IMAGE}

echo "Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE} \
    --platform managed \
    --region ${REGION} \
    --memory 2Gi \
    --timeout 300 \
    --concurrency 10 \
    --set-env-vars "GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_LOCATION=${REGION}" \
    --allow-unauthenticated \
    --project ${PROJECT_ID}

# Get service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region=${REGION} --project=${PROJECT_ID} --format='value(status.url)')

echo "=========================================="
echo "Deployment complete!"
echo "Service URL: ${SERVICE_URL}"
echo ""
echo "Test endpoints:"
echo "  Health: curl ${SERVICE_URL}/health"
echo "  Scrape: curl -X POST ${SERVICE_URL}/scrape_furniture_single -H 'Content-Type: application/json' -d '{\"url\": \"...\"}'"
echo "=========================================="
