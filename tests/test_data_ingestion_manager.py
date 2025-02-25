import unittest
import logging
import os
from tests.base_test import BaseTest
from src.ingestion.data_ingestion_manager import DataIngestionManager
from src.embeddings.embedding_manager import EmbeddingManager
from llama_index.core import Document, Settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestDataIngestionManager(BaseTest):
    needs_storage = True
    
    def setUp(self):
        super().setUp()
        self.data_ingestion_manager = DataIngestionManager(storage_path=self.TEST_STORAGE_PATH)
        logger.info("Setup completed. All components initialized.")
        
    def test_ingest_pdf(self):
        """Test PDF ingestion"""
        logger.info("Testing PDF ingestion...")
        # Skip if test PDF doesn't exist
        if not os.path.exists(self.TEST_PDF_PATH):
            self.skipTest(f"Test PDF file not found at {self.TEST_PDF_PATH}")
            
        self.data_ingestion_manager.ingest_pdf(self.TEST_PDF_PATH)
        logger.info("PDF ingestion test passed")
        
    def test_ingest_web_content(self):
        """Test web content ingestion"""
        logger.info("Testing web content ingestion...")
        self.data_ingestion_manager.ingest_web_content(self.TEST_URLS)
        logger.info("Web content ingestion test passed")

    def tearDown(self):
        """Clean up after tests"""
        logger.info("Cleaning up after DataIngestionManager test...")
        # Clean up test storage if it exists
        if os.path.exists(self.TEST_STORAGE_PATH):
            import shutil
            shutil.rmtree(self.TEST_STORAGE_PATH)
            logger.info(f"Removed test storage directory: {self.TEST_STORAGE_PATH}")
