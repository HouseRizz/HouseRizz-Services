"""
Firebase Storage utilities for uploading and managing furniture images.

Uploads images to Firebase Storage and returns public URLs for use in Firestore.
"""

import logging
from pathlib import Path
from typing import Optional

from google.cloud import storage

from services.virtual_staging_api.config import config

logger = logging.getLogger(__name__)


class StorageClient:
    """
    Firebase/Cloud Storage client for furniture images.
    
    Uploads images to a GCS bucket and returns public URLs.
    """
    
    _instance: Optional["StorageClient"] = None
    
    def __new__(cls) -> "StorageClient":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize Storage client."""
        if self._initialized:
            return
        
        logger.info("Initializing Cloud Storage Client")
        
        self.client = storage.Client(project=config.GOOGLE_CLOUD_PROJECT)
        self.bucket_name = f"{config.GOOGLE_CLOUD_PROJECT}-furniture"
        
        # Get or create bucket
        try:
            self.bucket = self.client.get_bucket(self.bucket_name)
            logger.info(f"Using existing bucket: {self.bucket_name}")
        except Exception:
            # Create bucket if it doesn't exist
            self.bucket = self.client.create_bucket(
                self.bucket_name,
                location=config.GOOGLE_CLOUD_LOCATION
            )
            # Make bucket public
            self.bucket.make_public(recursive=True, future=True)
            logger.info(f"Created new bucket: {self.bucket_name}")
        
        self._initialized = True
    
    def upload_image(self, local_path: str, destination_name: Optional[str] = None) -> str:
        """
        Upload an image to Cloud Storage.
        
        Args:
            local_path: Path to the local image file
            destination_name: Optional name in the bucket (defaults to filename)
            
        Returns:
            Public URL of the uploaded image
        """
        path = Path(local_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {local_path}")
        
        # Use filename if no destination specified
        blob_name = destination_name or f"furniture/{path.name}"
        
        # Upload the file
        blob = self.bucket.blob(blob_name)
        blob.upload_from_filename(str(path))
        
        # Make the blob public
        blob.make_public()
        
        public_url = blob.public_url
        logger.info(f"Uploaded: {path.name} → {public_url}")
        
        return public_url
    
    def delete_image(self, blob_name: str) -> bool:
        """Delete an image from storage."""
        try:
            blob = self.bucket.blob(blob_name)
            blob.delete()
            return True
        except Exception as e:
            logger.error(f"Failed to delete {blob_name}: {e}")
            return False
    
    def list_images(self, prefix: str = "furniture/") -> list[str]:
        """List all images in the bucket."""
        blobs = self.bucket.list_blobs(prefix=prefix)
        return [blob.public_url for blob in blobs]


def get_storage_client() -> StorageClient:
    """Get the Storage client singleton."""
    return StorageClient()
