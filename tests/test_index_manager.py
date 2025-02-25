"""
Test suite for IndexManager component.

This test suite verifies the functionality of the IndexManager class, which is responsible
for creating, saving, and loading vector indexes used in the RAG system.

Key Components Tested:
    - Index creation with documents
    - Index persistence (saving to disk)
    - Index loading from disk
    
Test Environment:
    - Uses a mock LLM (disabled) to avoid external API calls
    - Uses real embedding model for document vectorization
    - Uses temporary test storage that is cleaned up after tests
    
Dependencies:
    - BaseTest: Provides common test setup and teardown functionality
    - EmbeddingManager: For document vectorization
    - llama_index: For core indexing functionality
"""

from tests.base_test import BaseTest
from src.index.index_manager import IndexManager
from src.embeddings.embedding_manager import EmbeddingManager
from llama_index.core import Document, Settings
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
import logging

logger = logging.getLogger(__name__)

class TestIndexManager(BaseTest):
    """
    Test cases for IndexManager functionality.
    
    This test class inherits from BaseTest which provides:
        - Common test storage path management
        - Embedding model initialization
        - Automatic cleanup of test artifacts
    
    Attributes:
        needs_embeddings (bool): Flag indicating embedding model is required
        needs_storage (bool): Flag indicating storage handling is required
    """
    
    needs_embeddings = True
    needs_storage = True
    
    def setUp(self):
        """
        Test setup method that runs before each test.
        
        Sets up:
            1. Base test configuration through parent class
            2. LLM settings (disabled for testing)
            3. Index manager instance
            4. Embedding manager and model
            
        Note:
            LLM is explicitly disabled to prevent external API calls during testing
            using Settings.llm = None
        """
        super().setUp()
        # Disable LLM for tests to avoid external API calls
        Settings.llm = None
        Settings.context_window = 2048
        Settings.num_output = 256
        
        # Create an IndexManager instance for each test
        self.index_manager = IndexManager(self.TEST_STORAGE_PATH)
        # Use our EmbeddingManager for document vectorization
        self.embedding_manager = EmbeddingManager()
        self.embed_model = self.embedding_manager.load_embedding_model()
        
    def test_create_and_save_index(self):
        """
        Test index creation and persistence functionality.
        
        This test verifies that:
            1. An index can be created from a document
            2. The created index is valid (not None)
            3. The index can be saved to disk
            
        Test Flow:
            1. Create a test document
            2. Create an index with the document
            3. Verify index creation
            4. Save index to disk
            
        Note:
            Uses a simple test document to verify basic functionality.
            The actual content indexing is tested in integration tests.
        """
        logger.info("Testing index creation and saving...")
        documents = [Document(text="Test document")]
        index = self.index_manager.create_index(documents, self.embed_model)
        self.assertIsNotNone(index)
        self.index_manager.save_index()
        logger.info("Index creation and saving test passed")
        
    def test_load_index(self):
        """
        Test index loading functionality.
        
        This test verifies that:
            1. An index can be created and saved
            2. The saved index can be loaded back
            3. The loaded index is valid
            
        Test Flow:
            1. Create and save a test index
            2. Load the saved index
            3. Verify the loaded index
            
        Dependencies:
            - Requires successful execution of index creation and saving
            - Requires proper file system access
            
        Note:
            This test is dependent on the success of create_and_save_index
            functionality, making it an integration test of sorts.
        """
        logger.info("Testing index loading...")
        # First create and save an index
        documents = [Document(text="Test document for loading")]
        self.index_manager.create_index(documents, self.embed_model)
        self.index_manager.save_index()
        logger.info("Created and saved test index")
        
        # Now try to load it
        loaded_index = self.index_manager.load_index(self.embed_model)
        self.assertIsNotNone(loaded_index)
        logger.info("Index loading test passed")

if __name__ == '__main__':
    unittest.main()
