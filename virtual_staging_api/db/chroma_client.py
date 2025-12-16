"""
ChromaDB client wrapper for the furniture inventory vector database.

Provides a simple interface for:
- Initializing persistent local ChromaDB
- Adding furniture items with embeddings
- Querying similar items by embedding vector
"""

import chromadb
from chromadb.config import Settings
from typing import Optional
import logging

from backend.config import config

logger = logging.getLogger(__name__)


class ChromaClient:
    """Wrapper for ChromaDB operations on the furniture inventory."""
    
    _instance: Optional["ChromaClient"] = None
    
    def __new__(cls) -> "ChromaClient":
        """Singleton pattern to reuse the same client instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the ChromaDB client and collection."""
        if self._initialized:
            return
            
        logger.info(f"Initializing ChromaDB at {config.CHROMA_PERSIST_DIR}")
        
        self.client = chromadb.PersistentClient(
            path=config.CHROMA_PERSIST_DIR,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create the furniture collection
        self.collection = self.client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            metadata={"description": "Furniture inventory embeddings"}
        )
        
        self._initialized = True
        logger.info(f"ChromaDB collection '{config.CHROMA_COLLECTION_NAME}' ready with {self.collection.count()} items")
    
    def add_items(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
        documents: Optional[list[str]] = None
    ) -> None:
        """
        Add furniture items to the collection.
        
        Args:
            ids: Unique identifiers for each item
            embeddings: 1408-dim vectors from multimodalembedding@001
            metadatas: Dict with name, price, filepath, category for each item
            documents: Optional text descriptions
        """
        if not ids:
            logger.warning("No items to add")
            return
            
        # ChromaDB requires documents if not provided
        if documents is None:
            documents = [m.get("name", "") for m in metadatas]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
        
        logger.info(f"Added {len(ids)} items to collection")
    
    def query_similar(
        self,
        query_embedding: list[float],
        n_results: int = 5
    ) -> dict:
        """
        Query for similar furniture items.
        
        Args:
            query_embedding: 1408-dim vector to search with
            n_results: Number of results to return
            
        Returns:
            Dict with ids, distances, metadatas, and documents
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["metadatas", "documents", "distances"]
        )
        
        logger.info(f"Query returned {len(results['ids'][0])} results")
        return results
    
    def get_count(self) -> int:
        """Get the number of items in the collection."""
        return self.collection.count()
    
    def reset(self) -> None:
        """Delete all items from the collection. Use with caution!"""
        self.client.delete_collection(config.CHROMA_COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            metadata={"description": "Furniture inventory embeddings"}
        )
        logger.info("Collection reset")


# Convenience function to get the singleton instance
def get_chroma_client() -> ChromaClient:
    """Get the ChromaDB client singleton."""
    return ChromaClient()
