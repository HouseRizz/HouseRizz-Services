"""
Module 1: Data Ingestion (Inventory System)

This module handles:
- Traversing the furniture inventory directory
- Uploading images to Cloud Storage
- Generating 1408-dim embeddings using Vertex AI multimodalembedding@001
- Storing embeddings and metadata in Firestore

Usage:
    python -m services.virtual_staging_api.ingest
"""

import asyncio
import hashlib
import logging
import re
from pathlib import Path
from typing import Optional

import vertexai
from vertexai.vision_models import MultiModalEmbeddingModel, Image

from services.virtual_staging_api.config import config
from services.virtual_staging_api.db.firestore_client import get_vector_client
from services.virtual_staging_api.db.storage_client import get_storage_client
from services.virtual_staging_api.models import FurnitureItem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize Vertex AI
vertexai.init(
    project=config.GOOGLE_CLOUD_PROJECT,
    location=config.GOOGLE_CLOUD_LOCATION
)


def parse_furniture_metadata(filename: str) -> dict:
    """
    Extract furniture metadata from filename.
    
    The inventory images follow a naming pattern like:
    "calvary-sheesham-wood-arm-chair-in-provincial-teak-finish-...webp"
    
    This function extracts:
    - A human-readable name
    - A category (chair, sofa, table, etc.)
    
    Args:
        filename: The image filename (with or without extension)
        
    Returns:
        Dict with 'name', 'category', and 'id' keys
    
    Examples:
        >>> parse_furniture_metadata("calvary-sheesham-wood-arm-chair-in-teak.webp")
        {'name': 'Calvary Sheesham Wood Arm Chair', 'category': 'chair', 'id': 'abc123'}
    """
    # Remove extension
    name_part = Path(filename).stem
    
    # Generate a unique ID from the filename
    item_id = hashlib.md5(filename.encode()).hexdigest()[:12]
    
    # Split on hyphens and clean up
    parts = name_part.split("-")
    
    # Detect category from common furniture terms
    category = "furniture"
    category_keywords = {
        "chair": "chair",
        "armchair": "chair",
        "sofa": "sofa",
        "couch": "sofa",
        "table": "table",
        "desk": "table",
        "bed": "bed",
        "cabinet": "storage",
        "shelf": "storage",
        "lamp": "lighting",
        "rug": "decor"
    }
    
    for part in parts:
        if part.lower() in category_keywords:
            category = category_keywords[part.lower()]
            break
    
    # Build a readable name from the first meaningful parts
    # Stop at descriptors like "in", "with", "by"
    stop_words = {"in", "with", "by", "for", "and"}
    name_parts = []
    
    for part in parts:
        if part.lower() in stop_words:
            break
        # Skip random hash suffixes (typically 6+ alphanumeric at end)
        if len(part) >= 6 and part.isalnum() and not any(c.isalpha() for c in part[:3]):
            continue
        name_parts.append(part.capitalize())
    
    # Limit to first 6 words for readability
    name = " ".join(name_parts[:6])
    
    return {
        "id": item_id,
        "name": name or "Unknown Furniture",
        "category": category
    }


async def get_image_embedding(image_path: str) -> list[float]:
    """
    Generate a 1408-dimensional embedding vector for an image.
    
    Uses Vertex AI's multimodalembedding@001 model which produces
    embeddings suitable for semantic similarity search.
    
    Prompt Engineering Context:
    - The model analyzes visual features: color, texture, shape, style
    - Embeddings capture semantic meaning for furniture matching
    - No text prompt needed for image-only embedding
    
    Args:
        image_path: Absolute path to the image file
        
    Returns:
        List of 1408 floats representing the embedding vector
        
    Raises:
        FileNotFoundError: If the image doesn't exist
        Exception: If the Vertex AI API call fails
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    logger.info(f"Generating embedding for: {Path(image_path).name}")
    
    # Load the multimodal embedding model
    model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
    
    # Load the image
    image = Image.load_from_file(image_path)
    
    # Get embeddings - the model returns a MultiModalEmbeddingResponse
    # We want the image_embedding which is a 1408-dim vector
    embeddings = model.get_embeddings(
        image=image,
        dimension=1408  # Full dimension for best quality
    )
    
    return embeddings.image_embedding


async def ingest_single_item(
    image_path: Path,
    storage: object,
    default_price: float = 99.99
) -> tuple[str, list[float], dict]:
    """
    Process a single furniture image for ingestion.
    
    Args:
        image_path: Path to the image file
        storage: Cloud Storage client for uploads
        default_price: Default price if not specified in metadata
        
    Returns:
        Tuple of (id, embedding, metadata)
    """
    # Parse metadata from filename
    metadata = parse_furniture_metadata(image_path.name)
    
    # Upload image to Cloud Storage
    image_url = storage.upload_image(str(image_path))
    
    # Generate embedding
    embedding = await get_image_embedding(str(image_path))
    
    # Build full metadata for Firestore (with Cloud Storage URL)
    full_metadata = {
        "name": metadata["name"],
        "category": metadata["category"],
        "price": default_price,
        "filepath": image_url  # Cloud Storage URL instead of local path
    }
    
    return metadata["id"], embedding, full_metadata


async def ingest_inventory(
    inventory_dir: Optional[str] = None,
    reset_collection: bool = False
) -> int:
    """
    Traverse the inventory directory and ingest all furniture images.
    
    This is the main ingestion pipeline that:
    1. Scans the inventory directory for image files
    2. Generates embeddings for each image using Vertex AI
    3. Stores embeddings + metadata in ChromaDB
    
    Args:
        inventory_dir: Path to inventory images (defaults to config)
        reset_collection: If True, clear existing data before ingesting
        
    Returns:
        Number of items successfully ingested
    """
    inventory_path = Path(inventory_dir) if inventory_dir else config.INVENTORY_DIR
    
    if not inventory_path.exists():
        raise FileNotFoundError(f"Inventory directory not found: {inventory_path}")
    
    # Get Firestore client and Cloud Storage client
    db = get_vector_client()
    storage = get_storage_client()
    
    if reset_collection:
        logger.warning("Resetting collection - all existing data will be deleted")
        db.reset()
    
    # Find all image files
    image_extensions = {".webp", ".jpg", ".jpeg", ".png"}
    image_files = [
        f for f in inventory_path.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        logger.warning(f"No image files found in {inventory_path}")
        return 0
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Process all images
    ids = []
    embeddings = []
    metadatas = []
    
    for image_path in image_files:
        try:
            item_id, embedding, metadata = await ingest_single_item(image_path, storage)
            ids.append(item_id)
            embeddings.append(embedding)
            metadatas.append(metadata)
            logger.info(f"✓ Processed: {metadata['name']} → {metadata['filepath']}")
        except Exception as e:
            logger.error(f"✗ Failed to process {image_path.name}: {e}")
            continue
    
    # Batch add to Firestore
    if ids:
        db.add_items(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas
        )
        logger.info(f"Successfully ingested {len(ids)} items into Firestore")
    
    return len(ids)


def run_ingestion():
    """Synchronous wrapper to run the ingestion pipeline."""
    logger.info("=" * 50)
    logger.info("Starting Furniture Inventory Ingestion")
    logger.info(f"Project: {config.GOOGLE_CLOUD_PROJECT}")
    logger.info(f"Inventory: {config.INVENTORY_DIR}")
    logger.info("=" * 50)
    
    count = asyncio.run(ingest_inventory(reset_collection=True))
    
    logger.info("=" * 50)
    logger.info(f"Ingestion complete! Processed {count} items.")
    logger.info("=" * 50)
    
    return count


if __name__ == "__main__":
    run_ingestion()
