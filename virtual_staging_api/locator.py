
"""
Module 4: Locator Engine

This module handles object detection and localization within images using Gemini 2.0 Flash.
It identifies furniture items and returns their polygon boundaries.
"""

import logging
import json
from PIL import Image
import io

from google import genai
from google.genai import types

from services.virtual_staging_api.config import config

# Configure logging
logger = logging.getLogger(__name__)

# Initialize GenAI client
# Using us-central1 for Flash as it's generally available there
genai_client = genai.Client(
    vertexai=True,
    project=config.GOOGLE_CLOUD_PROJECT,
    location="us-central1",
)

async def localize_objects(
    image_bytes: bytes,
    target_objects: list[str] = None
) -> list[dict]:
    """
    Detect furniture objects in the image and return their polygon boundaries.
    
    Args:
        image_bytes: The image data
        target_objects: Optional list of specific objects to look for (e.g. ["sofa", "chair"])
    
    Returns:
        List of dicts with 'label' and 'polygon' (list of [ymin, xmin] coordinates)
    """
    logger.info("Starting object localization")
    
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        logger.error(f"Failed to open image for localization: {e}")
        raise ValueError("Invalid image data")

    # Construct prompt
    target_str = ", ".join(target_objects) if target_objects else "furniture items like sofas, chairs, tables, beds, rugs"
    
    prompt = f"""
    Detect the {target_str} in this image.
    
    For each furniture item found, provide:
    1. label: The type of furniture (e.g. "sofa", "chair")
    2. polygon: A DETAILED polygon outline that traces the EXACT silhouette/contour of the object.
       - The polygon should have at least 20-30 points to capture curves and edges accurately.
       - Points are in [x, y] format (NOT [y, x]).
       - All coordinates are normalized from 0.0 to 1.0 where (0,0) is top-left corner.
       - Trace the visible outline of the furniture as if you were drawing around it with a pen.
    
    Return ONLY the detected furniture items that are clearly visible in the image.
    """
    
    # Define schema for structured output
    response_schema = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "label": {"type": "STRING"},
                "polygon": {
                    "type": "ARRAY",
                    "items": {
                        "type": "ARRAY",
                        "items": {"type": "NUMBER"}
                    }
                }
            },
            "required": ["label", "polygon"]
        }
    }
    
    generate_config = types.GenerateContentConfig(
        temperature=0.5,
        top_p=0.95,
        max_output_tokens=4096,
        response_modalities=["TEXT"],
        response_mime_type="application/json",
        response_schema=response_schema
    )

    try:
        logger.info("Calling Gemini 2.0 Flash for object detection")
        response = genai_client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=[prompt, image],
            config=generate_config,
        )
        
        # Parse response
        result_text = response.text
        # Clean up code blocks if present (though response_mime_type usually handles this)
        if result_text.startswith("```json"):
            result_text = result_text[7:-3]
        elif result_text.startswith("```"):
            result_text = result_text[3:-3]
            
        objects = json.loads(result_text)
        logger.info(f"Detected {len(objects)} objects")
        
        return objects

    except Exception as e:
        logger.error(f"Localization failed: {e}")
        # Return empty list on failure rather than crashing if purely for UI
        return []
