"""
Module 5: Segmentor Engine

Wrapper for RAM-Grounded-SAM via Replicate API.
Provides object detection and segmentation for furniture highlighting.
"""

import os
import logging
import replicate

from services.virtual_staging_api.config import config

# Configure logging
logger = logging.getLogger(__name__)

# Set Replicate API token from environment or config
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN", "")


async def segment_image(image_bytes: bytes, use_sam_hq: bool = False) -> dict:
    """
    Segment all objects in an image using RAM-Grounded-SAM.
    
    Args:
        image_bytes: Image data
        use_sam_hq: Whether to use SAM-HQ for higher quality (slower)
    
    Returns:
        Dict with:
        - masked_img: URL to segmentation mask
        - rounding_box_img: URL to visualization with bounding boxes
        - json_data: Object detections with labels, boxes, and values
        - tags: Comma-separated list of detected object tags
    """
    logger.info("Running RAM-Grounded-SAM segmentation")
    
    if not REPLICATE_API_TOKEN:
        raise ValueError("REPLICATE_API_TOKEN not set")
    
    os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN
    
    try:
        # Create a file-like object from bytes
        import io
        image_file = io.BytesIO(image_bytes)
        image_file.name = "image.png"
        
        output = replicate.run(
            "idea-research/ram-grounded-sam:80a2aede4cf8e3c9f26e96c308d45b23c350dd36f1c381de790715007f1ac0ad",
            input={
                "input_image": image_file,
                "show_visualisation": True,
                "use_sam_hq": use_sam_hq,
            }
        )
        
        logger.info(f"Segmentation complete. Tags: {output.get('tags', 'N/A')}")
        
        return output
        
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        raise


def find_object_by_label(json_data: dict, target_labels: list[str]) -> dict | None:
    """
    Find an object in the detection results by label.
    
    Args:
        json_data: The json_data from segment_image output
        target_labels: List of labels to search for (e.g. ["couch", "sofa"])
    
    Returns:
        The detection dict with box, label, logit, value, or None if not found
    """
    masks = json_data.get("mask", [])
    
    for item in masks:
        label = item.get("label", "").lower()
        for target in target_labels:
            if target.lower() in label:
                return item
    
    return None
