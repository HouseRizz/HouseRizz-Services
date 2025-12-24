#!/bin/bash
set -e

PROJECT_ID="houserizz-481012"
REGION="us-central1"

echo "Deploying Product URL Crawler Functions..."

# url-extractor (extracts product URLs from category pages)
echo "Deploying url-extractor..."
cd url-extractor
gcloud functions deploy url-extractor \
    --gen2 \
    --runtime=python311 \
    --region=${REGION} \
    --source=. \
    --entry-point=start_extract_job \
    --trigger-http \
    --allow-unauthenticated \
    --set-env-vars "GOOGLE_CLOUD_PROJECT=${PROJECT_ID},TASK_LOCATION=${REGION},TASK_QUEUE=products-scraping" \
    --project=${PROJECT_ID}
cd ..

# job-list-api (lists crawl jobs)
echo "Deploying job-list-api..."
cd job-list-api
gcloud functions deploy job-list-api \
    --gen2 \
    --runtime=python311 \
    --region=${REGION} \
    --source=. \
    --entry-point=crawl_services_api \
    --trigger-http \
    --allow-unauthenticated \
    --set-env-vars "GOOGLE_CLOUD_PROJECT=${PROJECT_ID}" \
    --project=${PROJECT_ID}
cd ..

# url-extraction-poller (polls Hyperbrowser for extracted URLs)
echo "Deploying url-extraction-poller..."
cd url-extraction-poller
gcloud functions deploy url-extraction-poller \
    --gen2 \
    --runtime=python311 \
    --region=${REGION} \
    --source=. \
    --entry-point=poll_and_generate_csv \
    --trigger-http \
    --allow-unauthenticated \
    --set-env-vars "GOOGLE_CLOUD_PROJECT=${PROJECT_ID}" \
    --project=${PROJECT_ID}
cd ..

# scraper-trigger (triggers the product scraper)
echo "Deploying scraper-trigger..."
cd scraper-trigger
gcloud functions deploy scraper-trigger \
    --gen2 \
    --runtime=python311 \
    --region=${REGION} \
    --source=. \
    --entry-point=trigger_scrapper_job \
    --trigger-http \
    --allow-unauthenticated \
    --set-env-vars "GOOGLE_CLOUD_PROJECT=${PROJECT_ID}" \
    --project=${PROJECT_ID}
cd ..

echo "All crawler functions deployed!"
gcloud functions list --project=${PROJECT_ID}
