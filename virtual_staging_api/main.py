"""
Virtual Staging & E-commerce API

FastAPI application providing endpoints for:
- /health: Health check
- /ingest: Trigger inventory ingestion
- /select: Test selection engine (file path)
- /select/upload: Selection with base64 image upload
- /design: Full pipeline (Phase 2)

Run locally: python -m backend.main
Deploy to Cloud Run: ./deploy.sh
"""

import base64
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from services.virtual_staging_api.config import config
from services.virtual_staging_api.models import (
    HealthResponse,
    IngestResponse,
    SelectionRequest,
    SelectionRequestBase64,
    SelectionResponse,
    DesignRequest,
    DesignResponse,
    FurnitureItem,
    GenerateRequest,
    LocalizationRequest,
    LocalizationResponse,
    SegmentRequest,
    SegmentResponse,
    DetectedObject,
)
from services.virtual_staging_api.ingest import ingest_inventory
from services.virtual_staging_api.selector import select_furniture, select_furniture_from_bytes
from services.virtual_staging_api.db.firestore_client import get_vector_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup
    logger.info("=" * 50)
    logger.info("Virtual Staging API Starting")
    logger.info(f"Project: {config.GOOGLE_CLOUD_PROJECT}")
    logger.info(f"Location: {config.GOOGLE_CLOUD_LOCATION}")
    logger.info("=" * 50)
    
    # Initialize Firestore connection (lazy - don't fail startup)
    try:
        db = get_vector_client()
        count = db.get_count()
        logger.info(f"Firestore ready with {count} items")
    except Exception as e:
        logger.warning(f"Firestore initialization skipped: {e}")
        logger.info("Firestore will be initialized on first request")
    
    yield
    
    # Shutdown
    logger.info("Virtual Staging API Shutting Down")


# Create FastAPI app
app = FastAPI(
    title="Virtual Staging & E-commerce API",
    description="AI-powered room redesign with clickable furniture hotspots",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", version="0.1.0")


@app.post("/ingest", response_model=IngestResponse, tags=["Inventory"])
async def run_ingestion(reset: bool = False):
    """
    Trigger inventory ingestion.
    
    Scans the furniture_images directory and generates embeddings
    for all images, storing them in ChromaDB.
    
    Args:
        reset: If True, clear existing data before ingesting
        
    Returns:
        IngestResponse with count of processed items
    """
    try:
        count = await ingest_inventory(reset_collection=reset)
        return IngestResponse(
            success=True,
            items_processed=count,
            message=f"Successfully ingested {count} furniture items"
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.get("/inventory", tags=["Inventory"])
async def get_inventory_count():
    """Get the current inventory count from Firestore."""
    db = get_vector_client()
    return {"count": db.get_count()}


@app.get("/search", tags=["Search"])
async def search_furniture(query: str, top_k: int = 5):
    """
    Search furniture by text query (e.g., "white chair", "wooden table").
    
    This directly embeds the text query and searches the vector database.
    No room image required - useful for testing and simple searches.
    
    Args:
        query: Text description of desired furniture (e.g., "white chair")
        top_k: Number of results to return (default: 5)
        
    Returns:
        List of matching furniture items with similarity info
    """
    from services.virtual_staging_api.selector import get_text_embedding
    
    try:
        # Generate embedding for the text query
        query_embedding = await get_text_embedding(query)
        
        # Query Firestore for similar items
        db = get_vector_client()
        results = db.query_similar(
            query_embedding=query_embedding,
            n_results=top_k
        )
        
        # Format response
        items = []
        if results["ids"] and results["ids"][0]:
            for i, item_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i]
                items.append({
                    "id": item_id,
                    "name": metadata.get("name", "Unknown"),
                    "price": metadata.get("price", 0.0),
                    "category": metadata.get("category", "furniture"),
                    "filepath": metadata.get("filepath", "")
                })
        
        return {
            "query": query,
            "results": items,
            "count": len(items)
        }
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/select", response_model=SelectionResponse, tags=["Selection"])
async def select_furniture_endpoint(request: SelectionRequest):
    """
    Test the selection engine with a room image and vibe.
    
    This endpoint runs Modules 1-2:
    1. Analyzes the room image with Gemini 1.5 Pro
    2. Generates a semantic search query
    3. Queries ChromaDB for matching furniture
    4. Returns matched items with metadata
    
    Args:
        request: SelectionRequest with room_image_path, vibe_text, top_k
        
    Returns:
        SelectionResponse with search_query and selected_furniture
    """
    try:
        # Validate image path
        if not Path(request.room_image_path).exists():
            raise HTTPException(
                status_code=404,
                detail=f"Room image not found: {request.room_image_path}"
            )
        
        # Run selection pipeline
        search_query, furniture_items = await select_furniture(
            room_image_path=request.room_image_path,
            vibe_text=request.vibe_text,
            top_k=request.top_k
        )
        
        return SelectionResponse(
            search_query=search_query,
            selected_furniture=furniture_items
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Selection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Selection failed: {str(e)}")


@app.post("/select/upload", response_model=SelectionResponse, tags=["Selection"])
async def select_furniture_upload(request: SelectionRequestBase64):
    """
    Select furniture using a base64-encoded room image.
    
    This endpoint is designed for frontend integration where images
    are uploaded as base64 strings rather than file paths.
    
    Args:
        request: SelectionRequestBase64 with:
            - room_image_base64: Base64-encoded image data
            - mime_type: MIME type (e.g., "image/jpeg")
            - vibe_text: User's desired style
            - top_k: Number of items to return
        
    Returns:
        SelectionResponse with search_query and selected_furniture
    """
    try:
        # Decode base64 image
        try:
            image_bytes = base64.b64decode(request.room_image_base64)
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Invalid base64 image data"
            )
        
        # Run selection pipeline
        search_query, furniture_items = await select_furniture_from_bytes(
            image_bytes=image_bytes,
            mime_type=request.mime_type,
            vibe_text=request.vibe_text,
            top_k=request.top_k
        )
        
        return SelectionResponse(
            search_query=search_query,
            selected_furniture=furniture_items
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Selection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Selection failed: {str(e)}")


@app.post("/design", response_model=DesignResponse, tags=["Design"])
async def design_room(request: DesignRequest):
    """
    Full design pipeline: Select -> Compose -> Map.
    
    Note: Use /design/upload for frontend integration with base64 images.
    """
    raise HTTPException(
        status_code=400,
        detail="Use /design/upload endpoint with base64-encoded image for frontend integration."
    )


@app.post("/design/upload", tags=["Design"])
async def design_room_upload(request: dict):
    """
    Full design pipeline with base64 image upload.
    
    Pipeline:
    1. Analyze room to determine furniture needs
    2. Query database for matching furniture
    3. Generate redesigned room with Gemini image generation
    
    Args:
        room_image_base64: Base64-encoded room image
        mime_type: MIME type (default: image/jpeg)
        vibe_text: User's desired style
        
    Returns:
        Generated room with furniture URLs
    """
    from services.virtual_staging_api.composer import compose_room
    
    try:
        # Extract and validate request
        room_image_base64 = request.get("room_image_base64")
        mime_type = request.get("mime_type", "image/jpeg")
        vibe_text = request.get("vibe_text", "modern minimalist")
        
        if not room_image_base64:
            raise HTTPException(status_code=400, detail="room_image_base64 is required")
        
        # Decode base64 image
        try:
            image_bytes = base64.b64decode(room_image_base64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")
        
        # Run the full composition pipeline
        result = await compose_room(
            room_image_bytes=image_bytes,
            mime_type=mime_type,
            vibe_text=vibe_text
        )
        
        return {
            "success": True,
            "generated_image_url": result["generated_image_url"],
            "furniture_used": result["furniture_used"],
            "vibe": result["vibe"]
        }
        
    except ValueError as e:
        logger.error(f"Design failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Design failed: {e}")
        raise HTTPException(status_code=500, detail=f"Design generation failed: {str(e)}")


@app.post("/generate", tags=["Design"])
async def generate_room_manual(request: GenerateRequest):
    """
    Manually generate a staged room with specific furniture items.
    
    Skipping analysis and search steps. Directly uses provided furniture images.
    
    Args:
        request: GenerateRequest with base64 room and furniture images
        
    Returns:
        Generated room URL and metadata
    """
    from services.virtual_staging_api.composer import generate_staged_room
    
    try:
        # Decode room image
        try:
            room_image_bytes = base64.b64decode(request.room_image_base64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 room image data")
            
        # Prepare furniture items
        furniture_items = []
        for item in request.furniture_items:
            try:
                item_bytes = base64.b64decode(item.image_base64)
                furniture_items.append({
                    "name": item.name,
                    "image_bytes": item_bytes,
                    "mime_type": item.mime_type
                })
            except Exception:
                raise HTTPException(status_code=400, detail=f"Invalid base64 data for furniture: {item.name}")
        
        # Call generation directly
        _, generated_url = await generate_staged_room(
            room_image_bytes=room_image_bytes,
            room_mime_type=request.room_mime_type,
            furniture_items=furniture_items,
            vibe_text=request.vibe_text
        )
        
        return {
            "success": True,
            "generated_image_url": generated_url,
            "furniture_used": [
                {"name": item["name"], "type": "custom"} for item in furniture_items
            ],
            "vibe": request.vibe_text
        }
        
    except ValueError as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/detect-furniture", response_model=LocalizationResponse, tags=["Design"])
async def detect_furniture_endpoint(request: LocalizationRequest):
    """
    Detect furniture in an image and return polygon boundaries.
    
    Accepts either an image URL or base64 data.
    """
    import httpx
    from services.virtual_staging_api.locator import localize_objects
    
    try:
        image_bytes = None
        
        # Prioritize image_url
        if request.image_url:
            async with httpx.AsyncClient() as client:
                resp = await client.get(request.image_url)
                if resp.status_code != 200:
                    raise HTTPException(status_code=400, detail="Failed to download image from URL")
                image_bytes = resp.content
        elif request.image_base64:
            try:
                image_bytes = base64.b64decode(request.image_base64)
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid base64 image data")
        else:
            raise HTTPException(status_code=400, detail="Either image_url or image_base64 is required")
            
        # Run localization
        objects = await localize_objects(image_bytes, request.target_objects)
        
        return LocalizationResponse(
            objects=objects,
            count=len(objects)
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Localization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Localization failed: {str(e)}")


@app.post("/segment", response_model=SegmentResponse, tags=["Design"])
async def segment_image_endpoint(request: SegmentRequest):
    """
    Segment all objects in an image using RAM-Grounded-SAM.
    
    Returns mask URL and detected objects for furniture highlighting.
    """
    from services.virtual_staging_api.segmentor import segment_image
    
    try:
        # Decode base64 image
        try:
            image_bytes = base64.b64decode(request.image_base64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")
        
        # Run segmentation
        result = await segment_image(image_bytes, request.use_sam_hq)
        
        # Parse detections
        json_data = result.get("json_data", {})
        masks = json_data.get("mask", [])
        
        detected_objects = []
        for item in masks:
            if item.get("label") != "background":
                detected_objects.append(DetectedObject(
                    label=item.get("label", "unknown"),
                    box=item.get("box", [0, 0, 0, 0]),
                    confidence=item.get("logit", 0.0),
                    value=item.get("value", 0)
                ))
        
        return SegmentResponse(
            masked_img=result.get("masked_img", ""),
            visualization_img=result.get("rounding_box_img", ""),
            tags=result.get("tags", ""),
            objects=detected_objects
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")


@app.get("/furniture/{filepath:path}", tags=["Inventory"])
async def get_furniture_image(filepath: str):
    """
    Serve a furniture image by filepath.
    
    Useful for the frontend to display selected furniture items.
    
    Args:
        filepath: Path to the furniture image
        
    Returns:
        The image file
    """
    full_path = Path(filepath)
    
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(full_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=config.PORT,
        reload=True
    )
