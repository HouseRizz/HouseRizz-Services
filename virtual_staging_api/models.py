"""
Pydantic models for the Virtual Staging application.

Defines data validation schemas for:
- Furniture inventory items
- API requests and responses
- Coordinate mappings for clickable zones
"""

from typing import Optional
from pydantic import BaseModel, Field


class FurnitureItem(BaseModel):
    """Represents a furniture item in the inventory."""
    
    id: str = Field(..., description="Unique identifier for the furniture item")
    name: str = Field(..., description="Human-readable name of the furniture")
    price: float = Field(default=0.0, description="Price in user's currency")
    filepath: str = Field(..., description="Path to the furniture image file")
    category: str = Field(default="furniture", description="Category (chair, sofa, table, etc.)")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "chair-001",
                "name": "Calvary Sheesham Wood Arm Chair",
                "price": 299.99,
                "filepath": "/inventory/calvary-chair.webp",
                "category": "chair"
            }
        }
    }


class DesignRequest(BaseModel):
    """Request payload for the /design endpoint."""
    
    room_image_path: str = Field(..., description="Path to the room image file")
    vibe_text: str = Field(
        ..., 
        description="User's desired vibe/style for the room",
        examples=["cozy modern minimalist", "bohemian eclectic", "mid-century modern"]
    )
    top_k: int = Field(default=5, ge=1, le=10, description="Number of furniture items to select")


class DesignRequestBase64(BaseModel):
    """Request payload for /design/upload endpoint with base64 image."""
    
    room_image_base64: str = Field(..., description="Base64-encoded room image")
    mime_type: str = Field(default="image/jpeg", description="MIME type of the image")
    vibe_text: str = Field(..., description="User's desired vibe/style")
    top_k: int = Field(default=5, ge=1, le=10, description="Number of furniture items to use")


class SelectionRequest(BaseModel):
    """Request payload for the /select endpoint (testing selection engine)."""
    
    room_image_path: str = Field(..., description="Path to the room image file")
    vibe_text: str = Field(..., description="User's desired vibe/style")
    top_k: int = Field(default=5, ge=1, le=10, description="Number of items to return")


class SelectionRequestBase64(BaseModel):
    """Request payload for /select/upload endpoint with base64 image."""
    
    room_image_base64: str = Field(..., description="Base64-encoded room image")
    mime_type: str = Field(default="image/jpeg", description="MIME type of the image")
    vibe_text: str = Field(..., description="User's desired vibe/style")
    top_k: int = Field(default=5, ge=1, le=10, description="Number of items to return")


class BoundingBox(BaseModel):
    """Bounding box coordinates for a clickable zone."""
    
    ymin: float = Field(..., ge=0, le=1, description="Normalized top coordinate")
    xmin: float = Field(..., ge=0, le=1, description="Normalized left coordinate")
    ymax: float = Field(..., ge=0, le=1, description="Normalized bottom coordinate")
    xmax: float = Field(..., ge=0, le=1, description="Normalized right coordinate")


class ClickableZone(BaseModel):
    """Represents a clickable product zone in the generated image."""
    
    product: FurnitureItem
    bounding_box: BoundingBox


class DesignResponse(BaseModel):
    """Response payload for the /design endpoint."""
    
    generated_image_url: str = Field(..., description="URL/path to the generated room image")
    clickable_zones: list[ClickableZone] = Field(
        default_factory=list,
        description="List of clickable zones with product info and coordinates"
    )
    selected_furniture: list[FurnitureItem] = Field(
        default_factory=list,
        description="Full list of selected furniture items"
    )


class SelectionResponse(BaseModel):
    """Response payload for the /select endpoint."""
    
    search_query: str = Field(..., description="Generated search query from room analysis")
    selected_furniture: list[FurnitureItem] = Field(
        default_factory=list,
        description="Matched furniture items from inventory"
    )


class IngestResponse(BaseModel):
    """Response payload for the /ingest endpoint."""
    
    success: bool = Field(..., description="Whether ingestion was successful")
    items_processed: int = Field(..., description="Number of items ingested")
    message: str = Field(..., description="Status message")


class FurnitureItemInput(BaseModel):
    """Input model for manual furniture selection."""
    name: str = Field(default="Custom Furniture", description="Name of the furniture")
    image_base64: str = Field(..., description="Base64-encoded furniture image")
    mime_type: str = Field(default="image/png", description="MIME type of the furniture image")


class GenerateRequest(BaseModel):
    """Request payload for the /generate endpoint."""
    
    room_image_base64: str = Field(..., description="Base64-encoded room image")
    room_mime_type: str = Field(default="image/jpeg", description="MIME type of the room image")
    furniture_items: list[FurnitureItemInput] = Field(
        ..., 
        description="List of furniture items to place",
        min_items=1
    )
    vibe_text: str = Field(..., description="Desired style/vibe")


class LocalizationRequest(BaseModel):
    """Request payload for /detect-furniture endpoint."""
    image_base64: Optional[str] = Field(None, description="Base64-encoded image")
    image_url: Optional[str] = Field(None, description="Public URL of the image")
    target_objects: list[str] = Field(default=[], description="List of specific objects to detect")


class LocalizedObject(BaseModel):
    """Represents a detected object with its polygon boundary."""
    label: str = Field(..., description="Object label (e.g. sofa, chair)")
    polygon: list[list[float]] = Field(..., description="List of [ymin, xmin] coordinates")


class LocalizationResponse(BaseModel):
    """Response payload for /detect-furniture endpoint."""
    objects: list[LocalizedObject] = Field(..., description="List of detected objects")
    count: int = Field(..., description="Number of objects detected")


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(default="healthy")
    version: str = Field(default="0.1.0")


class SegmentRequest(BaseModel):
    """Request payload for /segment endpoint."""
    image_base64: str = Field(..., description="Base64-encoded image to segment")
    use_sam_hq: bool = Field(default=False, description="Use SAM-HQ for higher quality")


class DetectedObject(BaseModel):
    """A detected object with bounding box and label."""
    label: str = Field(..., description="Object label")
    box: list[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    confidence: float = Field(..., description="Detection confidence (logit)")
    value: int = Field(..., description="Mask value for this object")


class SegmentResponse(BaseModel):
    """Response payload for /segment endpoint."""
    masked_img: str = Field(..., description="URL to segmentation mask image")
    visualization_img: str = Field(..., description="URL to bounding box visualization")
    tags: str = Field(..., description="Comma-separated detected object tags")
    objects: list[DetectedObject] = Field(..., description="List of detected objects")


