"""
Module 3: Generation Engine (Composer)

This module implements the full design pipeline:
1. Analyze room image + user text to determine furniture needs
2. Query the vector database for each furniture piece  
3. Use Gemini image generation with room + furniture IMAGES to compose the redesigned room
4. Save the generated image to Cloud Storage

Usage:
    await compose_room(room_image_bytes, vibe_text)
"""

import asyncio
import base64
import io
import logging
import uuid
import os
from pathlib import Path
from typing import Optional
from PIL import Image

from google import genai
from google.genai import types

from services.virtual_staging_api.config import config
from services.virtual_staging_api.db.firestore_client import get_vector_client
from services.virtual_staging_api.db.storage_client import get_storage_client
from services.virtual_staging_api.selector import get_text_embedding

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize GenAI client with Vertex AI (uses Cloud Run service account)
genai_client = genai.Client(
    vertexai=True,
    project=config.GOOGLE_CLOUD_PROJECT,
    location="global",  # Global for image generation models
)


async def analyze_furniture_needs(
    room_image: Image.Image,
    vibe_text: str
) -> list[dict]:
    """
    Analyze a room image to determine what furniture is needed.
    """
    logger.info(f"Analyzing furniture needs for vibe: '{vibe_text}'")
    
    analysis_prompt = f"""You are an expert interior designer. Analyze this room image and the user's desired vibe.

USER'S DESIRED VIBE: {vibe_text}

Based on the room layout and the desired vibe, determine what furniture pieces would best fit this space.

Return a JSON array of furniture pieces needed, each with:
- "type": the category (chair, sofa, table, lamp, rug, etc.)
- "description": detailed search query for finding the right piece

IMPORTANT: Return ONLY the JSON array, no other text. Example:
[
    {{"type": "chair", "description": "warm wooden armchair with cushion"}},
    {{"type": "table", "description": "small round coffee table in wood"}}
]

Limit to 3-5 key pieces. Your response:"""

    response = genai_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[analysis_prompt, room_image],
    )
    
    # Parse JSON response
    import json
    try:
        result_text = response.text.strip()
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
        furniture_needs = json.loads(result_text)
        logger.info(f"Identified {len(furniture_needs)} furniture pieces needed")
        return furniture_needs
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse furniture needs: {e}")
        return [{"type": "furniture", "description": vibe_text}]


async def find_furniture_for_needs(
    furniture_needs: list[dict],
    top_k_per_type: int = 1
) -> list[dict]:
    """Query the database to find matching furniture for each need."""
    logger.info(f"Finding furniture for {len(furniture_needs)} needs")
    
    db = get_vector_client()
    all_furniture = []
    
    for need in furniture_needs:
        description = need.get("description", need.get("type", "furniture"))
        query_embedding = await get_text_embedding(description)
        
        results = db.query_similar(
            query_embedding=query_embedding,
            n_results=top_k_per_type
        )
        
        if results["ids"] and results["ids"][0]:
            for i, item_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i]
                
                # Get actual product type from metadata, or infer from name
                product_name = metadata.get("name", "Unknown")
                product_category = metadata.get("category", "")
                
                # Infer type from product name if category is generic
                inferred_type = product_category
                if not inferred_type or inferred_type == "furniture":
                    # Try to extract type from product name
                    name_lower = product_name.lower()
                    type_keywords = ["chair", "armchair", "sofa", "table", "bed", "lamp", "rug", "cabinet", "shelf", "desk", "stool", "ottoman", "bench"]
                    for kw in type_keywords:
                        if kw in name_lower:
                            inferred_type = kw
                            break
                    if not inferred_type:
                        inferred_type = need.get("type", "furniture")
                
                all_furniture.append({
                    "id": item_id,
                    "name": product_name,
                    "type": inferred_type,
                    "image_url": metadata.get("filepath", ""),
                    "price": metadata.get("price", 0.0),
                    "searched_for": description
                })
                logger.info(f"Found: {product_name} (type: {inferred_type}) for '{description}'")
    
    return all_furniture


def get_gcs_uri_from_public_url(public_url: str) -> str:
    """Convert public Cloud Storage URL to gs:// URI."""
    # https://storage.googleapis.com/bucket/path -> gs://bucket/path
    if public_url.startswith("https://storage.googleapis.com/"):
        path = public_url.replace("https://storage.googleapis.com/", "")
        return f"gs://{path}"
    return public_url


def get_mime_type(url: str) -> str:
    """Get MIME type from URL extension."""
    url_lower = url.lower()
    if url_lower.endswith(".png"):
        return "image/png"
    elif url_lower.endswith(".jpg") or url_lower.endswith(".jpeg"):
        return "image/jpeg"
    elif url_lower.endswith(".webp"):
        return "image/webp"
    return "image/png"


async def generate_staged_room(
    room_image_bytes: bytes,
    room_mime_type: str,
    furniture_items: list[dict],
    vibe_text: str
) -> tuple[bytes, str]:
    """
    Generate a staged room using Gemini 3 Pro image generation.
    
    Uses GCS URIs for furniture images directly.
    """
    logger.info(f"Generating staged room with {len(furniture_items)} furniture items")
    
    # Build content parts
    parts = []
    
    # Add furniture images from GCS
    for item in furniture_items:
        if item.get("image_url"):
            gcs_uri = get_gcs_uri_from_public_url(item["image_url"])
            mime_type = get_mime_type(item["image_url"])
            parts.append(types.Part.from_uri(
                file_uri=gcs_uri,
                mime_type=mime_type
            ))
            logger.info(f"Added furniture: {item['name']} from {gcs_uri}")
        elif item.get("image_bytes"):
            mime_type = item.get("mime_type", "image/png")
            parts.append(types.Part.from_bytes(
                data=item["image_bytes"],
                mime_type=mime_type
            ))
            # No logging of bytes for privacy/size
            logger.info(f"Added furniture: {item.get('name', 'Custom Item')} from bytes")
    
    # Add room image from bytes
    parts.append(types.Part.from_bytes(
        data=room_image_bytes,
        mime_type=room_mime_type
    ))
    
    # Add the text prompt
    furniture_names = ", ".join([item['name'] for item in furniture_items])
    prompt = f"""Generate a photorealistic interior design image of this room (the last image) redesigned with the furniture pieces shown in the other images.

STYLE: {vibe_text}

FURNITURE TO PLACE: {furniture_names}

Instructions:
- Place the selected furniture pieces naturally, ensuring perfect perspective and scale
- TRANSFORM the room's atmosphere (lighting, wall colors, flooring) to fully match the requested style
- PRESERVE structural elements (windows, doors, ceiling height) but UPGRADE finishes if they don't match the style
- Ensure the final image looks like a high-end editorial interior design photo

Generate a beautiful, photorealistic redesigned room."""

    parts.append(types.Part.from_text(text=prompt))
    
    # Build the content structure
    contents = [
        types.Content(
            role="user",
            parts=parts
        )
    ]
    
    # Configure generation
    generate_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        max_output_tokens=32768,
        response_modalities=["TEXT", "IMAGE"],
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
        ],
        image_config=types.ImageConfig(
            aspect_ratio="1:1",
            image_size="1K",
            output_mime_type="image/png",
        ),
    )
    
    logger.info(f"Generated parts count: {len(parts)}")
    for i, part in enumerate(parts):
        if part.text:
            logger.info(f"Part {i} is Text: {part.text[:50]}...")
        elif part.inline_data:
            logger.info(f"Part {i} is Inline Data: {len(part.inline_data.data)} bytes, mime: {part.inline_data.mime_type}")
        elif part.file_data:
            logger.info(f"Part {i} is File Data: {part.file_data.file_uri}")
    
    logger.info("Calling Gemini 3 Pro for image generation...")
    
    if not parts:
        msg = "No content parts generated for image generation request."
        logger.error(msg)
        raise ValueError(msg)

    # Generate content
    response = genai_client.models.generate_content(
        model="gemini-3-pro-image-preview",
        contents=contents,
        config=generate_config,
    )
    
    # Extract generated image
    generated_image_bytes = None
    for part in response.parts:
        if part.text:
            logger.info(f"Model text: {part.text[:200]}...")
        if hasattr(part, 'inline_data') and part.inline_data:
            generated_image_bytes = part.inline_data.data
            logger.info("Got image from inline_data")
            break
    
    if not generated_image_bytes:
        raise ValueError("Failed to generate image - no image in response")
    
    # Upload to Cloud Storage
    storage = get_storage_client()
    
    temp_path = f"/tmp/staged_room_{uuid.uuid4().hex}.png"
    with open(temp_path, "wb") as f:
        f.write(generated_image_bytes)
    
    public_url = storage.upload_image(
        temp_path,
        destination_name=f"generated/{uuid.uuid4().hex}.png"
    )
    
    Path(temp_path).unlink(missing_ok=True)
    
    logger.info(f"Generated room saved to: {public_url}")
    
    return generated_image_bytes, public_url


async def compose_room(
    room_image_bytes: bytes,
    mime_type: str,
    vibe_text: str
) -> dict:
    """
    Full composition pipeline: analyze → find furniture → generate.
    """
    logger.info("=" * 50)
    logger.info("Starting Room Composition Pipeline")
    logger.info(f"Vibe: {vibe_text}")
    logger.info("=" * 50)
    
    # Convert bytes to PIL Image for analysis
    room_image = Image.open(io.BytesIO(room_image_bytes))
    
    # Step 1: Analyze what furniture is needed
    furniture_needs = await analyze_furniture_needs(room_image, vibe_text)
    
    # Step 2: Find matching furniture from inventory
    furniture_items = await find_furniture_for_needs(furniture_needs)
    
    if not furniture_items:
        raise ValueError("No matching furniture found in inventory")
    
    # Step 3: Generate the staged room with furniture from GCS
    _, generated_url = await generate_staged_room(
        room_image_bytes, mime_type, furniture_items, vibe_text
    )
    
    logger.info("=" * 50)
    logger.info("Composition complete!")
    logger.info("=" * 50)
    
    return {
        "generated_image_url": generated_url,
        "furniture_used": furniture_items,
        "vibe": vibe_text
    }
