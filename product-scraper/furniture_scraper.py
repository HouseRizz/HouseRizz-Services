"""
Furniture Product Scraper

Scrapes furniture product URLs using LangChain's HyperbrowserLoader,
formats with LLM, and stores in both Firestore collections:
- products (for iOS app)
- furniture_inventory (with vector embeddings for AI search)

Usage:
    python furniture_scraper.py [--url URL]
    
API Endpoints:
    POST /scrape_furniture_single - Scrape single URL
    POST /scrape_furniture_csv - Process CSV file
"""

import asyncio
import json
import csv
import uuid
import os
import traceback
import hashlib
from datetime import datetime
from io import StringIO
from typing import Optional, Dict, List, Any

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

import vertexai
from vertexai.generative_models import GenerativeModel
from langchain_hyperbrowser import HyperbrowserLoader

from firebase_config import (
    get_firestore_client,
    download_image,
    upload_to_storage,
    generate_embedding,
    save_to_furniture_inventory,
    save_to_products_collection,
    logger,
    GOOGLE_CLOUD_PROJECT,
    GOOGLE_CLOUD_LOCATION
)

load_dotenv()

app = Flask(__name__)
CORS(app, origins="*", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"], allow_headers="*")

# Environment variables required:
# - HYPERBROWSER_API_KEY: API key for web scraping
# - GOOGLE_CLOUD_PROJECT: GCP project for Vertex AI (defaults to houserizz-481012)

# Validate required env vars
if not os.getenv("HYPERBROWSER_API_KEY"):
    logger.warning("HYPERBROWSER_API_KEY not set in environment")

# Initialize Vertex AI
vertexai.init(project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_CLOUD_LOCATION)

# Initialize Gemini 2.0 Flash model via Vertex AI
gemini_model = GenerativeModel('gemini-2.0-flash-exp')

# Furniture-specific prompt template
FURNITURE_PROMPT_TEMPLATE = """You are a JSON formatting assistant for furniture products. Your ONLY task is to output a single valid JSON object.

Format the following furniture product information into a valid JSON object with this exact structure:
{{
  "name": "[product name]",
  "description": "[detailed product description]",
  "category": "[MUST be one of: chair, sofa, table, bed, storage, lighting, decor, furniture]",
  "supplier": "[brand/manufacturer/seller name]",
  "address": "[supplier location, city, or 'Online' if not available]",
  "price": [numeric price without currency symbol, or null if not found],
  "imageURLs": ["[image URL 1]", "[image URL 2]", "[image URL 3]"],
  "modelURL": "[3D model URL if available, else null]",
  "materials": "[materials used: wood type, fabric, metal, etc.]",
  "dimensions": "[dimensions: LxWxH or similar if available]",
  "color": "[primary color or finish]",
  "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"]
}}

Here's the product information to format:

{content}

Rules:
1. If any field is missing, use null (for strings/numbers) or empty array (for arrays)
2. Do not include any markdown formatting
3. Do not include any text before or after the JSON
4. Your entire response must be valid JSON parseable by json.loads()
5. For "category", you MUST choose ONLY from: chair, sofa, table, bed, storage, lighting, decor, furniture
6. Extract up to 3 image URLs that are actual product images (not icons/logos)
7. Price should be a number only (e.g., 15999 not "₹15,999")
8. For furniture items like sofas, armchairs, recliners → use "sofa" or "chair" as appropriate
9. For cabinets, wardrobes, shelves, bookcases → use "storage"
10. For lamps, chandeliers, pendants → use "lighting"
11. For rugs, mirrors, vases, art → use "decor"
"""

# Valid furniture categories
VALID_CATEGORIES = {"chair", "sofa", "table", "bed", "storage", "lighting", "decor", "furniture"}


def normalize_category(category: str) -> str:
    """Normalize and validate furniture category."""
    if not category:
        return "furniture"
    
    category = category.lower().strip()
    
    # Map common variations
    category_map = {
        "armchair": "chair",
        "recliner": "chair",
        "dining chair": "chair",
        "office chair": "chair",
        "couch": "sofa",
        "sectional": "sofa",
        "loveseat": "sofa",
        "desk": "table",
        "dining table": "table",
        "coffee table": "table",
        "side table": "table",
        "console": "table",
        "wardrobe": "storage",
        "cabinet": "storage",
        "shelf": "storage",
        "bookcase": "storage",
        "dresser": "storage",
        "lamp": "lighting",
        "chandelier": "lighting",
        "pendant": "lighting",
        "sconce": "lighting",
        "rug": "decor",
        "carpet": "decor",
        "mirror": "decor",
        "vase": "decor",
        "art": "decor",
        "mattress": "bed",
        "bedframe": "bed"
    }
    
    if category in VALID_CATEGORIES:
        return category
    
    return category_map.get(category, "furniture")


def generate_item_id(name: str, supplier: str) -> str:
    """Generate a unique 12-char ID from name and supplier."""
    content = f"{name}-{supplier}".encode()
    return hashlib.md5(content).hexdigest()[:12]


async def scrape_furniture_product(
    url: str,
    product_uuid: str
) -> Dict[str, Any]:
    """
    Scrape a furniture product URL and store in both Firestore collections.
    
    Args:
        url: Product URL to scrape
        product_uuid: UUID for the product
        
    Returns:
        Dict with scraped product data
    """
    logger.info(f"Starting scrape for: {url}")
    
    # 1. Scrape the URL
    loader = HyperbrowserLoader(
        urls=url,
        api_key=os.environ["HYPERBROWSER_API_KEY"],
        operation="scrape",
        params={"scrape_options": {"formats": ["markdown"]}}
    )
    docs = loader.load()
    scraped_content = docs[0].page_content
    logger.info(f"Scraped {len(scraped_content)} chars from {url}")
    
    # 2. Format with Gemini LLM
    prompt = FURNITURE_PROMPT_TEMPLATE.format(content=scraped_content)
    response = gemini_model.generate_content(prompt)
    formatted_text = response.text
    
    # Parse JSON response
    try:
        formatted_json = json.loads(formatted_text)
    except json.JSONDecodeError:
        # Try stripping markdown
        text_content = formatted_text.replace('```json', '').replace('```', '').strip()
        formatted_json = json.loads(text_content)
    
    logger.info(f"Extracted: {formatted_json.get('name', 'Unknown')}")
    
    # 3. Normalize category
    raw_category = formatted_json.get("category", "furniture")
    category = normalize_category(raw_category)
    formatted_json["category"] = category
    
    # 4. Get first valid image URL for embedding
    image_urls = formatted_json.get("imageURLs", [])
    primary_image_url = None
    storage_image_url = None
    embedding = None
    
    if image_urls and len(image_urls) > 0:
        primary_image_url = image_urls[0]
        
        try:
            # Download image
            local_image_path = await download_image(primary_image_url)
            
            # Upload to Firebase Storage
            storage_path = f"furniture/{product_uuid}.jpg"
            storage_image_url = await upload_to_storage(local_image_path, storage_path)
            
            # Generate embedding
            embedding = await generate_embedding(local_image_path)
            logger.info(f"Generated {len(embedding)}-dim embedding")
            
            # Cleanup temp file
            os.unlink(local_image_path)
            
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            # Continue without embedding - we can still save to products
    
    # 5. Prepare data for products collection (iOS app format)
    current_time = datetime.utcnow().replace(microsecond=0).isoformat() + 'Z'
    
    products_data = {
        "id": product_uuid,
        "name": formatted_json.get("name", "Unknown Furniture"),
        "description": formatted_json.get("description"),
        "category": category,
        "supplier": formatted_json.get("supplier", "Unknown"),
        "address": formatted_json.get("address", "Online"),
        "price": formatted_json.get("price"),
        "imageURL1": image_urls[0] if len(image_urls) > 0 else None,
        "imageURL2": image_urls[1] if len(image_urls) > 1 else None,
        "imageURL3": image_urls[2] if len(image_urls) > 2 else None,
        "modelURL": formatted_json.get("modelURL"),
        # Additional metadata
        "materials": formatted_json.get("materials"),
        "dimensions": formatted_json.get("dimensions"),
        "color": formatted_json.get("color"),
        "keywords": formatted_json.get("keywords", []),
        "source": {
            "sourceUrl": url,
            "dateScraped": current_time
        },
        "dateCreated": current_time,
        "dateEdited": current_time
    }
    
    # 6. Save to products collection
    await save_to_products_collection(product_uuid, products_data)
    
    # 7. Save to furniture_inventory with embedding (if we have one)
    if embedding and storage_image_url:
        item_id = generate_item_id(
            formatted_json.get("name", "Unknown"),
            formatted_json.get("supplier", "Unknown")
        )
        
        await save_to_furniture_inventory(
            item_id=item_id,
            name=formatted_json.get("name", "Unknown Furniture"),
            price=float(formatted_json.get("price") or 0),
            filepath=storage_image_url,
            category=category,
            embedding=embedding
        )
        products_data["inventoryId"] = item_id
    
    return products_data


@app.route('/scrape_furniture_single', methods=['POST'])
def scrape_furniture_single():
    """
    Endpoint to scrape a single furniture product URL.
    
    Request JSON:
        {
            "url": "https://example.com/product",
            "productId": "optional-uuid"
        }
    
    Returns:
        {
            "success": true,
            "productId": "uuid",
            "data": {...},
            "message": "..."
        }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        url = data.get('url')
        if not url:
            return jsonify({'error': 'URL is required'}), 400
        
        product_uuid = data.get('productId', str(uuid.uuid4()))
        
        logger.info(f"Scraping furniture: {url} -> {product_uuid}")
        
        # Run async scraper
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(scrape_furniture_product(url, product_uuid))
            
            return jsonify({
                'success': True,
                'productId': product_uuid,
                'data': result,
                'message': 'Furniture product scraped successfully'
            })
            
        except Exception as e:
            error_msg = f"Scraping failed: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            
            return jsonify({
                'success': False,
                'error': error_msg,
                'message': 'Failed to scrape furniture product'
            }), 500
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Endpoint error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Server error occurred'
        }), 500


@app.route('/scrape_furniture_csv', methods=['POST'])
def scrape_furniture_csv():
    """
    Endpoint to process a CSV file with furniture product URLs.
    
    Request JSON:
        {
            "csv_url": "gs://bucket/file.csv or https://...",
            "max_items": 10  (optional, default 50)
        }
    
    CSV format:
        url,product_name,manufacturer
        https://...,Optional Name,Optional Brand
    
    Returns:
        {
            "success": true,
            "processed": 5,
            "failed": 1,
            "results": [...]
        }
    """
    try:
        data = request.get_json()
        if not data or 'csv_url' not in data:
            return jsonify({'error': 'csv_url is required'}), 400
        
        csv_url = data['csv_url']
        max_items = data.get('max_items', 50)
        
        # Download CSV
        if csv_url.startswith('gs://'):
            from google.cloud import storage
            bucket_name, blob_name = csv_url[5:].split('/', 1)
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            csv_content = blob.download_as_string().decode('utf-8')
        else:
            import requests as req
            response = req.get(csv_url)
            response.raise_for_status()
            csv_content = response.content.decode('utf-8')
        
        reader = csv.DictReader(StringIO(csv_content))
        products = list(reader)[:max_items]
        
        logger.info(f"Processing {len(products)} products from CSV")
        
        # Process each product
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        results = []
        processed = 0
        failed = 0
        
        try:
            for product in products:
                url = product.get('url')
                if not url:
                    continue
                
                product_uuid = product.get('uuid') or str(uuid.uuid4())
                
                try:
                    result = loop.run_until_complete(scrape_furniture_product(url, product_uuid))
                    results.append({
                        'url': url,
                        'productId': product_uuid,
                        'name': result.get('name'),
                        'status': 'success'
                    })
                    processed += 1
                    
                except Exception as e:
                    results.append({
                        'url': url,
                        'productId': product_uuid,
                        'status': 'failed',
                        'error': str(e)
                    })
                    failed += 1
                    
        finally:
            loop.close()
        
        return jsonify({
            'success': True,
            'processed': processed,
            'failed': failed,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"CSV processing error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'furniture-scraper',
        'version': '1.0.0'
    })


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--url':
        # Test mode: scrape single URL
        if len(sys.argv) < 3:
            print("Usage: python furniture_scraper.py --url <URL>")
            sys.exit(1)
        
        test_url = sys.argv[2]
        test_uuid = str(uuid.uuid4())
        
        print(f"Testing scrape for: {test_url}")
        result = asyncio.run(scrape_furniture_product(test_url, test_uuid))
        print(json.dumps(result, indent=2))
    else:
        # Server mode
        port = int(os.environ.get('PORT', 8080))
        app.run(host='0.0.0.0', port=port, debug=True)
