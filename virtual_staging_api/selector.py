"""
Module 2: Selection Engine (RAG-based Furniture Selector)

This module implements a RAG (Retrieval-Augmented Generation) pipeline:
1. Analyze room image + user vibe using Gemini 2.5 Flash
2. Generate a semantic search query
3. Embed query using multimodalembedding@001
4. Query Firestore for matching furniture
5. Return ranked results with metadata

Prompt Engineering Strategy:
- The room analysis prompt extracts style, color palette, and spatial constraints
- The search query is structured for optimal embedding similarity
- Results are filtered and ranked by relevance score
"""

import asyncio
import base64
import logging
from pathlib import Path
from typing import Optional

import vertexai
from google import genai
from google.genai import types
from vertexai.vision_models import MultiModalEmbeddingModel, Image

from services.virtual_staging_api.config import config
from services.virtual_staging_api.db.firestore_client import get_vector_client
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

# Initialize Google GenAI client
genai_client = genai.Client(
    vertexai=True,
    project=config.GOOGLE_CLOUD_PROJECT,
    location=config.GOOGLE_CLOUD_LOCATION
)


async def analyze_room(room_image_path: str, vibe: str) -> str:
    """
    Analyze a room image and user vibe to generate a furniture search query.
    
    Uses Gemini 1.5 Pro for vision understanding. The model analyzes:
    - Room dimensions and layout
    - Existing furniture and decor
    - Lighting conditions
    - Color palette
    - User's requested style/vibe
    
    Prompt Engineering Details:
    - We ask for a specific search query format to maximize embedding quality
    - The query should describe ideal furniture attributes
    - Include style, material, color preferences based on room analysis
    - Output is concise (1-2 sentences) for optimal embedding
    
    Args:
        room_image_path: Path to the room image file
        vibe: User's desired style/vibe description
        
    Returns:
        A search query string like "Mid-century modern walnut wood armchair 
        with warm tones to complement neutral living room"
    """
    if not Path(room_image_path).exists():
        raise FileNotFoundError(f"Room image not found: {room_image_path}")
    
    logger.info(f"Analyzing room with vibe: '{vibe}'")
    
    # Read image and encode as base64
    with open(room_image_path, "rb") as f:
        image_bytes = f.read()
    
    # Determine MIME type
    suffix = Path(room_image_path).suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp"
    }
    mime_type = mime_types.get(suffix, "image/jpeg")
    
    # Construct the analysis prompt
    analysis_prompt = f"""You are an expert interior designer analyzing a room for virtual staging.

TASK: Analyze this room image and the user's desired vibe to generate a furniture search query.

USER'S DESIRED VIBE: {vibe}

Analyze the room for:
1. Room type (living room, bedroom, office, etc.)
2. Current style and color palette
3. Lighting (natural, warm, cool)
4. Space available for furniture
5. Existing elements to complement

Then generate a SINGLE search query (1-2 sentences) describing the ideal furniture piece(s) 
that would match this room and vibe. Include:
- Furniture type (chair, sofa, table, etc.)
- Material preferences (wood, leather, fabric, etc.)
- Color/finish that would complement the space
- Style descriptors (modern, traditional, bohemian, etc.)

OUTPUT FORMAT:
Just the search query, nothing else. Example:
"Warm-toned walnut wood armchair with cream upholstery in mid-century modern style"

Your search query:"""

    # Create the content parts
    contents = [
        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        types.Part.from_text(analysis_prompt)
    ]
    
    # Generate response using Gemini 2.5 Flash
    response = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: genai_client.models.generate_content(
            model=f"projects/{config.GOOGLE_CLOUD_PROJECT}/locations/{config.GOOGLE_CLOUD_LOCATION}/publishers/google/models/{config.GEMINI_MODEL}",
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=200,
            )
        )
    )
    
    search_query = response.text.strip().strip('"')
    logger.info(f"Generated search query: {search_query}")
    
    return search_query


async def analyze_room_from_bytes(image_bytes: bytes, mime_type: str, vibe: str) -> str:
    """
    Analyze a room image from bytes (for frontend uploads).
    
    Uses Gemini 2.5 Flash for vision understanding.
    
    Args:
        image_bytes: Raw image bytes
        mime_type: MIME type of the image (e.g., "image/jpeg")
        vibe: User's desired style/vibe description
        
    Returns:
        A search query string describing ideal furniture
    """
    logger.info(f"Analyzing room from bytes with vibe: '{vibe}'")
    
    # Construct the analysis prompt
    analysis_prompt = f"""You are an expert interior designer analyzing a room for virtual staging.

TASK: Analyze this room image and the user's desired vibe to generate a furniture search query.

USER'S DESIRED VIBE: {vibe}

Analyze the room for:
1. Room type (living room, bedroom, office, etc.)
2. Current style and color palette
3. Lighting (natural, warm, cool)
4. Space available for furniture
5. Existing elements to complement

Then generate a SINGLE search query (1-2 sentences) describing the ideal furniture piece(s) 
that would match this room and vibe. Include:
- Furniture type (chair, sofa, table, etc.)
- Material preferences (wood, leather, fabric, etc.)
- Color/finish that would complement the space
- Style descriptors (modern, traditional, bohemian, etc.)

OUTPUT FORMAT:
Just the search query, nothing else. Example:
"Warm-toned walnut wood armchair with cream upholstery in mid-century modern style"

Your search query:"""

    # Create the content parts
    contents = [
        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        types.Part.from_text(analysis_prompt)
    ]
    
    # Generate response using Gemini 2.5 Flash
    response = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: genai_client.models.generate_content(
            model=f"projects/{config.GOOGLE_CLOUD_PROJECT}/locations/{config.GOOGLE_CLOUD_LOCATION}/publishers/google/models/{config.GEMINI_MODEL}",
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=200,
            )
        )
    )
    
    search_query = response.text.strip().strip('"')
    logger.info(f"Generated search query: {search_query}")
    
    return search_query


async def get_text_embedding(text: str) -> list[float]:
    """
    Generate a 1408-dimensional embedding for a text query.
    
    Uses the same multimodalembedding@001 model as image embeddings,
    ensuring consistent embedding space for cross-modal search.
    
    Args:
        text: The search query to embed
        
    Returns:
        List of 1408 floats representing the text embedding
    """
    logger.info(f"Generating text embedding for query")
    
    model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
    
    # Get text embeddings in the same space as image embeddings
    embeddings = model.get_embeddings(
        contextual_text=text,
        dimension=1408
    )
    
    return embeddings.text_embedding


async def get_image_query_embedding(image_path: str) -> list[float]:
    """
    Generate embedding for a room image to find similar furniture.
    
    This allows image-to-image search in addition to text-to-image.
    
    Args:
        image_path: Path to the query image
        
    Returns:
        1408-dim embedding vector
    """
    model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
    image = Image.load_from_file(image_path)
    
    embeddings = model.get_embeddings(
        image=image,
        dimension=1408
    )
    
    return embeddings.image_embedding


async def select_furniture(
    room_image_path: str,
    vibe_text: str,
    top_k: int = 5
) -> tuple[str, list[FurnitureItem]]:
    """
    Main selection pipeline: analyze room, search inventory, return matches.
    
    This is the core RAG function that:
    1. Uses Gemini 1.5 Pro to understand the room and desired vibe
    2. Generates a semantic search query
    3. Embeds the query using multimodalembedding@001
    4. Searches ChromaDB for similar furniture embeddings
    5. Returns ranked furniture items with full metadata
    
    Args:
        room_image_path: Path to the room image to analyze
        vibe_text: User's desired style/vibe (e.g., "cozy modern minimalist")
        top_k: Number of furniture items to return (1-10)
        
    Returns:
        Tuple of (search_query, list[FurnitureItem])
        
    Raises:
        FileNotFoundError: If room image doesn't exist
        ValueError: If no furniture found in inventory
    """
    logger.info("=" * 50)
    logger.info("Starting furniture selection pipeline")
    logger.info(f"Room: {room_image_path}")
    logger.info(f"Vibe: {vibe_text}")
    logger.info("=" * 50)
    
    # Step 1: Analyze room and generate search query
    search_query = await analyze_room(room_image_path, vibe_text)
    
    # Step 2: Embed the search query
    query_embedding = await get_text_embedding(search_query)
    
    # Step 3: Query Firestore for similar furniture
    db = get_vector_client()
    
    if db.get_count() == 0:
        raise ValueError("No furniture in inventory. Run ingestion first: python -m backend.ingest")
    
    results = db.query_similar(
        query_embedding=query_embedding,
        n_results=top_k
    )
    
    # Step 4: Convert results to FurnitureItem objects
    furniture_items = []
    
    if results["ids"] and results["ids"][0]:
        for i, item_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i] if results["distances"] else 0
            
            item = FurnitureItem(
                id=item_id,
                name=metadata.get("name", "Unknown"),
                price=metadata.get("price", 0.0),
                filepath=metadata.get("filepath", ""),
                category=metadata.get("category", "furniture")
            )
            furniture_items.append(item)
            
            logger.info(f"Match {i+1}: {item.name} (distance: {distance:.4f})")
    
    logger.info(f"Selected {len(furniture_items)} furniture items")
    
    return search_query, furniture_items


async def select_furniture_from_bytes(
    image_bytes: bytes,
    mime_type: str,
    vibe_text: str,
    top_k: int = 5
) -> tuple[str, list[FurnitureItem]]:
    """
    Selection pipeline for frontend uploads (base64 images).
    
    Same as select_furniture but accepts raw bytes instead of file path.
    
    Args:
        image_bytes: Raw image bytes
        mime_type: MIME type of the image
        vibe_text: User's desired style/vibe
        top_k: Number of furniture items to return
        
    Returns:
        Tuple of (search_query, list[FurnitureItem])
    """
    logger.info("=" * 50)
    logger.info("Starting furniture selection (from bytes)")
    logger.info(f"Vibe: {vibe_text}")
    logger.info("=" * 50)
    
    # Step 1: Analyze room and generate search query
    search_query = await analyze_room_from_bytes(image_bytes, mime_type, vibe_text)
    
    # Step 2: Embed the search query
    query_embedding = await get_text_embedding(search_query)
    
    # Step 3: Query Firestore for similar furniture
    db = get_vector_client()
    
    if db.get_count() == 0:
        raise ValueError("No furniture in inventory. Run ingestion first.")
    
    results = db.query_similar(
        query_embedding=query_embedding,
        n_results=top_k
    )
    
    # Step 4: Convert results to FurnitureItem objects
    furniture_items = []
    
    if results["ids"] and results["ids"][0]:
        for i, item_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i] if results["distances"] else 0
            
            item = FurnitureItem(
                id=item_id,
                name=metadata.get("name", "Unknown"),
                price=metadata.get("price", 0.0),
                filepath=metadata.get("filepath", ""),
                category=metadata.get("category", "furniture")
            )
            furniture_items.append(item)
            
            logger.info(f"Match {i+1}: {item.name} (distance: {distance:.4f})")
    
    logger.info(f"Selected {len(furniture_items)} furniture items")
    
    return search_query, furniture_items


async def demo_selection():
    """Demo function to test the selection pipeline."""
    # Use a furniture image as a stand-in for a room image (for testing)
    inventory_dir = config.INVENTORY_DIR
    
    # Get first image from inventory for demo
    images = list(inventory_dir.glob("*.webp"))
    if not images:
        logger.error("No images found in inventory. Add some first!")
        return
    
    room_image = str(images[0])
    vibe = "cozy modern minimalist with warm wooden tones"
    
    query, items = await select_furniture(room_image, vibe, top_k=3)
    
    print("\n" + "=" * 50)
    print("SELECTION RESULTS")
    print("=" * 50)
    print(f"Search Query: {query}")
    print(f"\nSelected Furniture ({len(items)} items):")
    for i, item in enumerate(items, 1):
        print(f"  {i}. {item.name} (${item.price}) - {item.category}")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(demo_selection())
