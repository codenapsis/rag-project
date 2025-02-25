"""
Base Test Configuration for RAG System.

This module provides the foundational test infrastructure for the RAG system,
implementing common setup, teardown, and utility functions used across all test cases.
It serves as a base class that standardizes test environment configuration and
resource management.

Key Features:
    - Common test environment configuration
    - Shared resource management
    - Standard cleanup procedures
    - Unified logging setup
    - Test storage handling
    
Test Environment Management:
    - LLM configuration (disabled for tests)
    - Storage path standardization
    - Embedding model initialization
    - Resource cleanup protocols
    
Usage:
    class TestYourComponent(BaseTest):
        needs_embeddings = True  # If your test needs embeddings
        needs_storage = True     # If your test needs storage
        
        def setUp(self):
            super().setUp()      # Initialize base configuration
            # Your specific setup code
            
        def test_your_feature(self):
            # Your test code using self.TEST_STORAGE_PATH, etc.
"""

import unittest
import logging
import os
import shutil
from llama_index.core import Settings
from src.embeddings.embedding_manager import EmbeddingManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseTest(unittest.TestCase):
    """
    Base test class providing common functionality for all RAG system tests.
    
    This class implements shared testing infrastructure, including:
        - Standard test environment setup
        - Resource initialization and management
        - Common cleanup procedures
        - Shared test utilities
        
    Class Attributes:
        TEST_STORAGE_PATH (str): Standard path for test storage
        TEST_PDF_PATH (str): Standard path for test PDF files
        TEST_URLS (list): Standard test URLs for web content
        
    Instance Flags:
        needs_embeddings (bool): Flag to enable embedding model initialization
        needs_storage (bool): Flag to enable storage directory management
        
    Usage:
        Inherit from this class and set flags as needed:
        needs_embeddings = True  # If embeddings are required
        needs_storage = True     # If storage is required
    """
    
    @classmethod
    def setUpClass(cls):
        """
        Class-level setup that runs once before all tests.
        
        Configures:
            1. Global test settings
            2. LLM configuration (disabled)
            3. Standard test paths
            4. Common test data
            
        Note:
            This method runs once per test class, not per test method.
            Use for expensive setup operations that can be shared across tests.
        """
        # Configure basic settings
        Settings.llm = None
        Settings.context_window = 2048
        Settings.num_output = 256
        
        # Set up common test paths
        cls.TEST_STORAGE_PATH = "test_storage"
        cls.TEST_PDF_PATH = os.path.join(os.getcwd(), "pdfs", "test.pdf")
        cls.TEST_URLS = ["http://example.com"]
        
    def setUp(self):
        """
        Instance-level setup that runs before each test method.
        
        Performs:
            1. Logging initialization
            2. Embedding model setup (if needed)
            3. Storage directory creation (if needed)
            
        Features:
            - Conditional embedding model initialization
            - Dynamic storage directory management
            - Detailed logging of setup process
            
        Note:
            This method runs before each test method.
            Use for test-specific setup that can't be shared.
        """
        logger.info(f"Setting up test: {self.__class__.__name__}")
        
        # Create embedding manager and model if needed
        if hasattr(self, 'needs_embeddings') and self.needs_embeddings:
            self.embedding_manager = EmbeddingManager()
            self.embed_model = self.embedding_manager.load_embedding_model()
            logger.info("Embedding model loaded")
            
        # Create test storage directory if needed
        if hasattr(self, 'needs_storage') and self.needs_storage:
            os.makedirs(self.TEST_STORAGE_PATH, exist_ok=True)
            logger.info(f"Created test storage at {self.TEST_STORAGE_PATH}")
    
    def tearDown(self):
        """
        Instance-level cleanup that runs after each test method.
        
        Performs:
            1. Storage cleanup (if used)
            2. Resource deallocation
            3. Logging of cleanup activities
            
        Features:
            - Conditional storage cleanup
            - Resource deallocation
            - Detailed cleanup logging
            
        Note:
            This method runs after each test method.
            Ensures clean test isolation by removing test artifacts.
        """
        logger.info(f"Cleaning up after test: {self.__class__.__name__}")
        
        # Clean up test storage if it was created
        if hasattr(self, 'needs_storage') and self.needs_storage:
            if os.path.exists(self.TEST_STORAGE_PATH):
                shutil.rmtree(self.TEST_STORAGE_PATH)
                logger.info(f"Removed test storage: {self.TEST_STORAGE_PATH}")
    
    @classmethod
    def tearDownClass(cls):
        """
        Class-level cleanup that runs once after all tests.
        
        Performs:
            1. Final storage cleanup
            2. Shared resource deallocation
            3. Global state cleanup
            
        Note:
            This method runs once per test class.
            Use for final cleanup of shared resources.
        """
        # Clean up any remaining test artifacts
        if os.path.exists(cls.TEST_STORAGE_PATH):
            shutil.rmtree(cls.TEST_STORAGE_PATH)
            logger.info(f"Cleaned up test storage in tearDownClass") 