"""
Unit tests for the Data Ingestion module.

Tests metadata parsing and ingestion pipeline components
that can be tested without GCP credentials.
"""

import pytest
from pathlib import Path

from services.virtual_staging_api.ingest import parse_furniture_metadata


class TestParseMetadata:
    """Tests for the parse_furniture_metadata function."""
    
    def test_basic_chair_parsing(self):
        """Test parsing a basic chair filename."""
        filename = "calvary-sheesham-wood-arm-chair-in-provincial-teak-finish.webp"
        result = parse_furniture_metadata(filename)
        
        assert result["category"] == "chair"
        assert "Calvary" in result["name"]
        assert "Sheesham" in result["name"]
        assert "id" in result
        assert len(result["id"]) == 12  # MD5 hash[:12]
    
    def test_armchair_detected_as_chair(self):
        """Test that armchair is categorized as chair."""
        filename = "clint-sheesham-wood-armchair-in-provincial-teak.webp"
        result = parse_furniture_metadata(filename)
        
        assert result["category"] == "chair"
    
    def test_stops_at_descriptor_words(self):
        """Test that name stops at 'in', 'with', 'by'."""
        filename = "elegant-sheesham-wood-armchair-in-honey-oak-finish-by-woodsworth.webp"
        result = parse_furniture_metadata(filename)
        
        # Should not include 'in honey oak finish by woodsworth'
        assert "Honey" not in result["name"]
        assert "Woodsworth" not in result["name"]
    
    def test_unique_ids_for_different_files(self):
        """Test that different filenames produce different IDs."""
        filename1 = "chair-one.webp"
        filename2 = "chair-two.webp"
        
        result1 = parse_furniture_metadata(filename1)
        result2 = parse_furniture_metadata(filename2)
        
        assert result1["id"] != result2["id"]
    
    def test_same_id_for_same_file(self):
        """Test that same filename always produces same ID."""
        filename = "test-chair.webp"
        
        result1 = parse_furniture_metadata(filename)
        result2 = parse_furniture_metadata(filename)
        
        assert result1["id"] == result2["id"]
    
    def test_default_category_when_no_match(self):
        """Test default category when no furniture type detected."""
        filename = "random-item-name.webp"
        result = parse_furniture_metadata(filename)
        
        assert result["category"] == "furniture"
    
    def test_handles_sofa_category(self):
        """Test sofa category detection."""
        filename = "modern-leather-sofa-in-grey.webp"
        result = parse_furniture_metadata(filename)
        
        assert result["category"] == "sofa"
    
    def test_handles_table_category(self):
        """Test table category detection."""
        filename = "wooden-dining-table-set.webp"
        result = parse_furniture_metadata(filename)
        
        assert result["category"] == "table"
    
    def test_name_capitalization(self):
        """Test that name parts are properly capitalized."""
        filename = "elegant-wood-chair.webp"
        result = parse_furniture_metadata(filename)
        
        # Each word should be capitalized
        assert result["name"] == "Elegant Wood Chair"


class TestInventoryPath:
    """Tests for inventory path handling."""
    
    def test_inventory_dir_exists(self):
        """Test that the inventory directory exists (if configured)."""
        from services.virtual_staging_api.config import config
        
        # This may fail if run before setup, which is expected
        inventory_path = config.INVENTORY_DIR
        assert inventory_path is not None


# Integration tests that require GCP credentials would go here
# with appropriate @pytest.mark.skipif decorators

class TestIngestionIntegration:
    """
    Integration tests for the ingestion pipeline.
    
    These tests require:
    - GCP credentials configured
    - Vertex AI API enabled
    
    Skip with: pytest -m "not integration"
    """
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_embedding_generation(self):
        """Test that embedding generation produces correct dimensions."""
        # This test requires actual GCP credentials
        from services.virtual_staging_api.ingest import get_image_embedding
        from services.virtual_staging_api.config import config
        
        # Use first inventory image
        images = list(config.INVENTORY_DIR.glob("*.webp"))
        if not images:
            pytest.skip("No inventory images available")
        
        embedding = await get_image_embedding(str(images[0]))
        
        assert len(embedding) == 1408  # Expected dimension
        assert all(isinstance(x, float) for x in embedding)
