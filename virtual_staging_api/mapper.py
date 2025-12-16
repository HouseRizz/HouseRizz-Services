"""
Module 4: Mapping Engine (Coordinate Extraction)

This module handles extracting bounding box coordinates from generated images:
1. Takes a generated room image
2. Uses Gemini 1.5 Flash to identify furniture locations
3. Returns JSON with bounding boxes for each product

Note: This is a stub for Phase 2 implementation.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


async def get_coordinates(
    generated_image_path: str,
    product_list: list[dict]
) -> dict:
    """
    Extract bounding box coordinates for products in a generated image.
    
    Uses Gemini 1.5 Flash for fast coordinate extraction with a structured
    output prompt.
    
    Prompt Engineering Strategy (for future implementation):
    - Prompt: "Identify the bounding box [ymin, xmin, ymax, xmax] for the 
      [Product Name] in this image. Return JSON."
    - Use normalized coordinates (0-1) for responsive overlay
    - Handle cases where product may not be visible
    
    Args:
        generated_image_path: Path to the generated room image
        product_list: List of dicts with product info (id, name, etc.)
        
    Returns:
        Dict mapping product IDs to bounding boxes:
        {
            "product_id_1": {
                "ymin": 0.2, "xmin": 0.1, "ymax": 0.6, "xmax": 0.4
            },
            ...
        }
        
    Raises:
        NotImplementedError: This module is not yet implemented
    """
    logger.warning("Module 4 (mapper) not yet implemented")
    
    raise NotImplementedError(
        "Module 4 (Mapping Engine) is planned for Phase 2. "
        "This module will use Gemini 1.5 Flash for fast "
        "bounding box extraction."
    )


async def validate_coordinates(
    coordinates: dict,
    image_width: int,
    image_height: int
) -> dict:
    """
    Validate and denormalize coordinate values.
    
    Converts normalized (0-1) coordinates to pixel coordinates
    and validates they are within image bounds.
    
    Args:
        coordinates: Dict of normalized coordinates
        image_width: Image width in pixels
        image_height: Image height in pixels
        
    Returns:
        Dict with pixel coordinates
    """
    raise NotImplementedError("Coordinate validation not yet implemented")
