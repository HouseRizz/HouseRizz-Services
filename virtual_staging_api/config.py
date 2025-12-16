"""
Configuration module for the Virtual Staging application.

Loads environment variables and provides centralized configuration access.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration loaded from environment variables."""
    
    # Google Cloud
    GOOGLE_CLOUD_PROJECT: str = os.getenv("GOOGLE_CLOUD_PROJECT", "houserizz-481012")
    GOOGLE_CLOUD_LOCATION: str = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    
    # Vertex AI Models - Using Gemini 2.5 Flash for all operations
    EMBEDDING_MODEL: str = "multimodalembedding@001"
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-preview-05-20")
    
    # Embedding dimensions
    EMBEDDING_DIM: int = 1408
    
    # Firestore Collection
    FIRESTORE_COLLECTION: str = os.getenv("FIRESTORE_COLLECTION", "furniture_inventory")
    
    # Cloud Run / Service Config
    PORT: int = int(os.getenv("PORT", "8080"))
    SERVICE_URL: str = os.getenv("SERVICE_URL", "http://localhost:8080")
    
    # Paths (config.py is at services/virtual_staging_api/config.py)
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent  # houserizz/
    INVENTORY_DIR: Path = PROJECT_ROOT / "furniture_images"
    
    @classmethod
    def validate(cls) -> None:
        """Validate that required configuration is present."""
        if not cls.GOOGLE_CLOUD_PROJECT:
            raise ValueError("GOOGLE_CLOUD_PROJECT environment variable is required")
        
        if not cls.INVENTORY_DIR.exists():
            raise ValueError(f"Inventory directory not found: {cls.INVENTORY_DIR}")


# Singleton instance
config = Config()
