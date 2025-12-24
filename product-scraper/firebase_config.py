"""
Firebase and GCP Configuration for HouseRizz Furniture Scraper.

Provides centralized initialization for:
- Firebase Admin SDK (Firestore)
- Cloud Storage client
- Vertex AI for embeddings
"""

import os
import logging
from pathlib import Path
from typing import Optional
import tempfile

import firebase_admin
from firebase_admin import credentials, firestore, storage
import vertexai
from google.cloud import storage as gcs_storage
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from vertexai.vision_models import MultiModalEmbeddingModel, Image
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# HouseRizz Project Configuration
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "houserizz-481012")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
STORAGE_BUCKET = os.getenv("STORAGE_BUCKET", "houserizz-481012.firebasestorage.app")

# Initialize Firebase Admin SDK
_firebase_app = None
_firestore_client = None
_storage_client = None


def _init_firebase():
    """Initialize Firebase Admin SDK for HouseRizz project."""
    global _firebase_app, _firestore_client
    
    if _firebase_app is not None:
        return
    
    try:
        # Try to find credentials file
        cred_paths = [
            os.path.join(os.path.dirname(__file__), 'houserizz-481012-firebase.json'),
            os.path.join(os.path.dirname(__file__), '..', '..', 'houserizz-481012-7388c153c534.json'),
            os.getenv('GOOGLE_APPLICATION_CREDENTIALS', '')
        ]
        
        cred_path = None
        for path in cred_paths:
            if path and os.path.exists(path):
                cred_path = path
                break
        
        if cred_path:
            cred = credentials.Certificate(cred_path)
            _firebase_app = firebase_admin.initialize_app(cred, {
                'storageBucket': STORAGE_BUCKET
            })
            logger.info(f"Firebase initialized with credentials from: {cred_path}")
        else:
            # Use default credentials (for Cloud Run)
            _firebase_app = firebase_admin.initialize_app(options={
                'storageBucket': STORAGE_BUCKET
            })
            logger.info("Firebase initialized with default credentials")
        
        _firestore_client = firestore.client()
        logger.info("Firestore client ready")
        
    except Exception as e:
        logger.error(f"Failed to initialize Firebase: {e}")
        raise


def get_firestore_client():
    """Get the Firestore client singleton."""
    if _firestore_client is None:
        _init_firebase()
    return _firestore_client


def get_storage_bucket():
    """Get Firebase Storage bucket."""
    if _firebase_app is None:
        _init_firebase()
    return storage.bucket()


def init_vertex_ai():
    """Initialize Vertex AI for embeddings."""
    vertexai.init(
        project=GOOGLE_CLOUD_PROJECT,
        location=GOOGLE_CLOUD_LOCATION
    )
    logger.info(f"Vertex AI initialized for project: {GOOGLE_CLOUD_PROJECT}")


async def download_image(url: str) -> str:
    """
    Download an image from URL to a temporary file.
    
    Args:
        url: Image URL to download
        
    Returns:
        Path to the temporary file
    """
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Determine extension from content-type
        content_type = response.headers.get('content-type', 'image/jpeg')
        ext_map = {
            'image/jpeg': '.jpg',
            'image/png': '.png',
            'image/webp': '.webp',
            'image/gif': '.gif'
        }
        ext = ext_map.get(content_type, '.jpg')
        
        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        for chunk in response.iter_content(chunk_size=8192):
            temp_file.write(chunk)
        temp_file.close()
        
        logger.info(f"Downloaded image to: {temp_file.name}")
        return temp_file.name
        
    except Exception as e:
        logger.error(f"Failed to download image from {url}: {e}")
        raise


async def upload_to_storage(local_path: str, destination_path: str) -> str:
    """
    Upload a file to Firebase Storage.
    
    Args:
        local_path: Path to local file
        destination_path: Destination path in bucket (e.g., 'furniture/image.jpg')
        
    Returns:
        Public URL of the uploaded file
    """
    try:
        bucket = get_storage_bucket()
        blob = bucket.blob(destination_path)
        
        # Upload file
        blob.upload_from_filename(local_path)
        
        # Make public and get URL
        blob.make_public()
        public_url = blob.public_url
        
        logger.info(f"Uploaded to: {public_url}")
        return public_url
        
    except Exception as e:
        logger.error(f"Failed to upload {local_path}: {e}")
        raise


async def generate_embedding(image_path: str) -> list[float]:
    """
    Generate a 1408-dimensional embedding for an image.
    
    Args:
        image_path: Path to the image file (local)
        
    Returns:
        List of 1408 floats representing the embedding
    """
    try:
        init_vertex_ai()
        
        model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
        image = Image.load_from_file(image_path)
        
        embeddings = model.get_embeddings(
            image=image,
            dimension=1408
        )
        
        logger.info(f"Generated embedding for: {Path(image_path).name}")
        return embeddings.image_embedding
        
    except Exception as e:
        logger.error(f"Failed to generate embedding for {image_path}: {e}")
        raise


async def save_to_furniture_inventory(
    item_id: str,
    name: str,
    price: float,
    filepath: str,
    category: str,
    embedding: list[float]
) -> None:
    """
    Save a furniture item to the furniture_inventory collection with vector embedding.
    
    Args:
        item_id: Unique document ID
        name: Furniture name
        price: Price in currency
        filepath: Cloud Storage URL of the image
        category: Furniture category
        embedding: 1408-dim embedding vector
    """
    try:
        db = get_firestore_client()
        
        # Convert to Firestore Vector
        embedding_vector = Vector(embedding)
        
        doc_data = {
            "name": name,
            "price": price,
            "filepath": filepath,
            "category": category,
            "embedding": embedding_vector
        }
        
        db.collection("furniture_inventory").document(item_id).set(doc_data)
        logger.info(f"Saved to furniture_inventory: {item_id}")
        
    except Exception as e:
        logger.error(f"Failed to save to furniture_inventory: {e}")
        raise


async def save_to_products_collection(
    product_id: str,
    product_data: dict
) -> None:
    """
    Save a product to the products collection (iOS app format).
    
    Args:
        product_id: Unique document ID (UUID)
        product_data: Product data matching HRProduct schema
    """
    try:
        db = get_firestore_client()
        db.collection("products").document(product_id).set(product_data)
        logger.info(f"Saved to products: {product_id}")
        
    except Exception as e:
        logger.error(f"Failed to save to products: {e}")
        raise
