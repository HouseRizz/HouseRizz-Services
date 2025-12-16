"""
Firestore Vector Storage for furniture inventory.

Uses Firestore's native vector search capability with find_nearest()
for similarity search on embeddings.

Requires:
- google-cloud-firestore >= 2.14.0
- A Firestore database in Native mode
"""

import logging
from typing import Optional

from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure

from services.virtual_staging_api.config import config

logger = logging.getLogger(__name__)


class FirestoreVectorClient:
    """
    Firestore-based vector storage for furniture inventory.
    
    Uses Firestore's native vector search for similarity queries.
    Each furniture item is stored as a document with:
    - id: document ID
    - name: furniture name
    - price: price
    - filepath: image path
    - category: furniture category
    - embedding: Vector field for similarity search
    """
    
    _instance: Optional["FirestoreVectorClient"] = None
    
    def __new__(cls) -> "FirestoreVectorClient":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize Firestore client and collection."""
        if self._initialized:
            return
        
        logger.info("Initializing Firestore Vector Client")
        
        self.db = firestore.Client(project=config.GOOGLE_CLOUD_PROJECT)
        self.collection = self.db.collection("furniture_inventory")
        
        self._initialized = True
        logger.info("Firestore Vector Client ready")
    
    def add_items(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
        documents: Optional[list[str]] = None
    ) -> None:
        """
        Add furniture items with embeddings to Firestore.
        
        Args:
            ids: Unique identifiers for each item
            embeddings: 1408-dim vectors from multimodalembedding@001
            metadatas: Dict with name, price, filepath, category
            documents: Optional text descriptions (stored as 'description')
        """
        if not ids:
            logger.warning("No items to add")
            return
        
        batch = self.db.batch()
        
        for i, item_id in enumerate(ids):
            doc_ref = self.collection.document(item_id)
            
            # Convert embedding to Firestore Vector
            embedding_vector = Vector(embeddings[i])
            
            doc_data = {
                "name": metadatas[i].get("name", "Unknown"),
                "price": metadatas[i].get("price", 0.0),
                "filepath": metadatas[i].get("filepath", ""),
                "category": metadatas[i].get("category", "furniture"),
                "embedding": embedding_vector,
            }
            
            if documents and i < len(documents):
                doc_data["description"] = documents[i]
            
            batch.set(doc_ref, doc_data)
        
        batch.commit()
        logger.info(f"Added {len(ids)} items to Firestore")
    
    def query_similar(
        self,
        query_embedding: list[float],
        n_results: int = 5
    ) -> dict:
        """
        Query for similar furniture items using vector search.
        
        Uses Firestore's find_nearest() for cosine similarity search.
        
        Args:
            query_embedding: 1408-dim vector to search with
            n_results: Number of results to return
            
        Returns:
            Dict with ids, distances, metadatas (matching ChromaDB format)
        """
        # Convert to Firestore Vector
        query_vector = Vector(query_embedding)
        
        # Perform vector similarity search
        results = self.collection.find_nearest(
            vector_field="embedding",
            query_vector=query_vector,
            distance_measure=DistanceMeasure.COSINE,
            limit=n_results
        ).get()
        
        # Format results to match ChromaDB response structure
        ids = []
        metadatas = []
        distances = []
        documents = []
        
        for doc in results:
            data = doc.to_dict()
            ids.append(doc.id)
            metadatas.append({
                "name": data.get("name", "Unknown"),
                "price": data.get("price", 0.0),
                "filepath": data.get("filepath", ""),
                "category": data.get("category", "furniture")
            })
            # Cosine distance (lower is better)
            distances.append(data.get("distance", 0.0))
            documents.append(data.get("description", ""))
        
        logger.info(f"Query returned {len(ids)} results")
        
        return {
            "ids": [ids],
            "metadatas": [metadatas],
            "distances": [distances],
            "documents": [documents]
        }
    
    def get_count(self) -> int:
        """Get the number of items in the collection."""
        # Use aggregation for efficient count
        count_query = self.collection.count()
        results = count_query.get()
        
        for result in results:
            return result[0].value
        
        return 0
    
    def reset(self) -> None:
        """Delete all items from the collection. Use with caution!"""
        docs = self.collection.limit(500).stream()
        deleted = 0
        
        for doc in docs:
            doc.reference.delete()
            deleted += 1
        
        # Recursively delete if there are more
        if deleted >= 500:
            self.reset()
        
        logger.info("Firestore collection reset")
    
    def get_item(self, item_id: str) -> Optional[dict]:
        """Get a single item by ID."""
        doc = self.collection.document(item_id).get()
        if doc.exists:
            return doc.to_dict()
        return None


def get_vector_client() -> FirestoreVectorClient:
    """Get the Firestore vector client singleton."""
    return FirestoreVectorClient()
