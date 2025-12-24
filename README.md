# HouseRizz Services

Microservices for furniture product data extraction and processing.

## Services Overview

| Service | Type | Description |
|---------|------|-------------|
| `product-scraper` | Cloud Run | Scrapes individual furniture product pages using Hyperbrowser + Gemini AI |
| `job-status-api` | Cloud Run | REST API for monitoring scraping job status |
| `product-merge-api` | Cloud Function | Merges duplicate product records using OpenAI |
| `product-url-crawler/` | Cloud Functions | Pipeline for extracting product URLs from category pages |
| `virtual_staging_api` | Cloud Run | AI-powered virtual staging for furniture placement |

## Product URL Crawler Pipeline

```
url-extractor → url-extraction-poller → scraper-trigger → product-scraper
```

| Sub-service | Description |
|-------------|-------------|
| `url-extractor` | Starts URL extraction job from a category page URL |
| `url-extraction-poller` | Polls Hyperbrowser API for extracted URLs, saves CSV |
| `scraper-trigger` | Triggers the product scraper with extracted URLs |
| `job-list-api` | Lists and manages crawl jobs |

## Deployment

Each service has its own deployment script or uses gcloud commands:

```bash
# Deploy product-scraper (Cloud Run)
cd product-scraper && ./deploy.sh

# Deploy crawler functions
cd product-url-crawler && ./deploy_all.sh

# Deploy job-status-api
cd job-status-api && ./deploy.sh
```

## Secrets Management

All API keys are stored in GCP Secret Manager:
- `HYPERBROWSER_API_KEY` - For web scraping
- `OPENAI_API_KEY` - For product merging

Services fetch secrets at runtime using `google-cloud-secret-manager`.
