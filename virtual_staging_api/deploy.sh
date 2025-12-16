#!/bin/bash
# Deploy Virtual Staging API to Cloud Run
# Run from: services/virtual_staging_api/

set -e

# Get script directory (for running from any location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PROJECT_ID=${1:-$(gcloud config get-value project)}
SERVICE_NAME="virtual-staging-api"
REGION=${2:-"us-central1"}
MEMORY="1Gi"
CPU="1"

if [ -z "$PROJECT_ID" ]; then
    echo "❌ Error: Project ID required"
    echo "Usage: ./deploy.sh <project-id> [region]"
    exit 1
fi

echo "🚀 Deploying Virtual Staging API to Cloud Run"
echo "  Project: $PROJECT_ID"
echo "  Service: $SERVICE_NAME"
echo "  Region: $REGION"
echo "  Source: $SCRIPT_DIR"
echo ""

# Enable required APIs
echo "📡 Enabling APIs..."
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    aiplatform.googleapis.com \
    firestore.googleapis.com \
    storage.googleapis.com \
    --project="$PROJECT_ID" \
    --quiet

# Build and deploy from service directory
echo "🔨 Building and deploying..."
gcloud run deploy "$SERVICE_NAME" \
    --source=. \
    --platform=managed \
    --region="$REGION" \
    --memory="$MEMORY" \
    --cpu="$CPU" \
    --timeout="300" \
    --max-instances="10" \
    --min-instances="0" \
    --set-env-vars="GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_LOCATION=${REGION}" \
    --allow-unauthenticated \
    --quiet \
    --project="$PROJECT_ID"

echo "✅ Deployment complete!"
echo ""

SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" \
    --platform=managed \
    --region="$REGION" \
    --format='value(status.url)' \
    --project="$PROJECT_ID")

echo "🌐 Service URL: $SERVICE_URL"
echo ""
echo "Test endpoints:"
echo "  curl $SERVICE_URL/health"
echo "  curl $SERVICE_URL/inventory"
echo "  curl \"$SERVICE_URL/search?query=wooden+chair\""
echo ""
echo "API Docs:"
echo "  $SERVICE_URL/docs"
