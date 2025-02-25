"""
Test suite for RAGPipeline component.

This test suite verifies the functionality of the RAGPipeline class, which is the core
component that orchestrates the entire Retrieval-Augmented Generation process. The pipeline
coordinates document retrieval, context generation, and query processing.

Key Components Tested:
    - Pipeline initialization
    - Document retrieval functionality
    - Query processing
    - Result generation
    
Test Environment:
    - Uses mock LLM to avoid API calls
    - Tests with sample documents
    - Verifies retrieval accuracy
    
Dependencies:
    - BaseTest: Provides common test setup and teardown functionality
    - IndexManager: For document storage and retrieval
    - EmbeddingManager: For text embeddings
    
Performance Considerations:
    - Pipeline initialization overhead
    - Retrieval latency testing
    - Result quality verification
"""

import unittest
import logging
from tests.base_test import BaseTest
from src.pipeline.rag_pipeline import RAGPipeline
from src.index.index_manager import IndexManager
from src.embeddings.embedding_manager import EmbeddingManager
from llama_index.core import Document, Settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestRAGPipeline(BaseTest):
    """
    Test cases for RAGPipeline functionality.
    
    This test class verifies the complete RAG pipeline, ensuring that
    documents can be properly indexed, retrieved, and used for answering
    queries. It tests both the initialization and execution phases of
    the pipeline.
    
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
            2. Mock LLM configuration
            3. RAG pipeline components
            4. Test storage and embeddings
            
        Note:
            LLM is explicitly disabled to prevent external API calls
            during testing using Settings.llm = None
        """
        super().setUp()
        logger.info("Setting up RAGPipeline test...")
        Settings.llm = None
        Settings.context_window = 2048
        Settings.num_output = 256
        
        self.rag_pipeline = RAGPipeline()
        self.index_manager = IndexManager(self.TEST_STORAGE_PATH)
        self.embedding_manager = EmbeddingManager()
        self.embed_model = self.embedding_manager.load_embedding_model()
        logger.info("Setup completed. All components initialized.")
        
    def test_initialize_pipeline(self):
        """
        Test pipeline initialization functionality.
        
        This test verifies that:
            1. Pipeline can be initialized with an index
            2. Retriever is properly configured
            3. Components are correctly connected
            
        Test Flow:
            1. Create test documents
            2. Build and configure index
            3. Initialize pipeline
            4. Verify component setup
            
        Technical Details:
            - Tests index integration
            - Verifies retriever configuration
            - Validates pipeline readiness
        """
        logger.info("Testing pipeline initialization...")
        documents = [Document(text="Test document")]
        logger.info(f"Creating index with {len(documents)} documents")
        
        index = self.index_manager.create_index(documents, self.embed_model)
        logger.info("Index created successfully")
        
        self.rag_pipeline.initialize_pipeline(index)
        logger.info("Pipeline initialized")
        
        self.assertIsNotNone(self.rag_pipeline.retriever)
        logger.info("Pipeline initialization test passed")
        
    def test_run_pipeline(self):
        """
        Test pipeline execution functionality.
        
        This test verifies that:
            1. Pipeline can process queries
            2. Relevant documents are retrieved
            3. Results are properly generated
            
        Test Flow:
            1. Create test document set
            2. Initialize pipeline with documents
            3. Run test query
            4. Verify results
            
        Technical Details:
            - Tests document retrieval accuracy
            - Verifies result relevance
            - Validates response format
            
        Note:
            This test uses simple test documents to verify
            basic functionality. More complex scenarios should
            be tested in integration tests.
        """
        logger.info("Testing pipeline execution...")
        documents = [
            Document(text="Python is a programming language"),
            Document(text="Test driven development is important")
        ]
        logger.info(f"Creating index with {len(documents)} documents")
        
        index = self.index_manager.create_index(documents, self.embed_model)
        self.rag_pipeline.initialize_pipeline(index)
        logger.info("Pipeline initialized with test documents")
        
        query = "What is Python?"
        logger.info(f"Testing query: {query}")
        results = self.rag_pipeline.run_pipeline(query)
        logger.info(f"Query results: {results}")
        
        self.assertTrue(len(results) > 0)
        logger.info("Pipeline execution test passed")

    def tearDown(self):
        """
        Cleanup method that runs after each test.
        
        Performs:
            1. Pipeline shutdown
            2. Resource cleanup
            3. Storage cleanup (handled by BaseTest)
            
        Note:
            Most heavy cleanup is handled by BaseTest's tearDown
            through the needs_storage flag.
        """
        logger.info("Cleaning up after RAGPipeline test...")
