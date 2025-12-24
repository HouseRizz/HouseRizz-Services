#!/bin/bash
set -e

PROJECT_ID="houserizz-481012"
REGION="us-central1"
SERVICE_NAME="process-status"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "Building ${SERVICE_NAME} for linux/amd64..."

# Build locally with linux/amd64 platform (required for Cloud Run on Mac M1/ARM)
docker build --platform linux/amd64 -t ${IMAGE} .

echo "Pushing to GCR..."
docker push ${IMAGE}

echo "Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE} \
    --platform managed \
    --region ${REGION} \
    --memory 512Mi \
    --allow-unauthenticated \
    --project ${PROJECT_ID}

echo "Service URL: $(gcloud run services describe ${SERVICE_NAME} --region=${REGION} --project=${PROJECT_ID} --format='value(status.url)')"
