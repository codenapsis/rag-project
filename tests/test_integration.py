"""
Test suite for RAG System Integration.

This test suite verifies the end-to-end functionality of the RAG system, ensuring that
all components work together correctly. It tests the complete workflow from document
ingestion through query processing, validating the system's overall behavior.

Key Components Tested:
    - Document Processing Pipeline
    - Embedding Generation System
    - Index Creation and Management
    - Query Processing and Response Generation
    - Component Integration and Communication
    
Test Environment:
    - Uses mock LLM to avoid API calls
    - Tests with controlled document set
    - Verifies cross-component interaction
    - Uses temporary test storage
    
Dependencies:
    - RAGPipeline: Core pipeline orchestration
    - IndexManager: Document storage and retrieval
    - EmbeddingManager: Vector embeddings
    - DocumentProcessor: Text processing
    - ContentHandler: Raw content processing
    
Performance Considerations:
    - Full system initialization overhead
    - Cross-component communication latency
    - Resource cleanup requirements
"""

import unittest
import logging
import os
import shutil
from src.pipeline.rag_pipeline import RAGPipeline
from src.index.index_manager import IndexManager
from src.embeddings.embedding_manager import EmbeddingManager
from src.documents.document_processor import DocumentProcessor
from src.content.content_handler import ContentHandler
from llama_index.core import Document, Settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestIntegration(unittest.TestCase):
    """
    Integration test cases for the complete RAG system.
    
    This test class verifies that all system components can work together
    harmoniously, testing the complete workflow from document ingestion
    to query response generation.
    
    Key Integration Points:
        - Document processing to embedding generation
        - Embedding storage and retrieval
        - Query processing and context generation
        - Response synthesis and formatting
        
    Test Prerequisites:
        - All component dependencies available
        - Sufficient system resources
        - Temporary storage permissions
    """
    
    def setUp(self):
        """
        Test setup method that runs before each test.
        
        Sets up:
            1. Mock LLM configuration
            2. Test storage environment
            3. All system components
            4. Integration test configuration
            
        Components Initialized:
            - RAG Pipeline
            - Index Manager
            - Embedding Manager
            - Document Processor
            
        Note:
            Creates a clean test environment for each test case
            to ensure isolation and reproducibility.
        """
        logger.info("Setting up integration test...")
        
        # Disable LLM for tests
        Settings.llm = None
        Settings.context_window = 2048
        Settings.num_output = 256
        
        # Create test storage directory
        self.test_storage_path = "test_storage"
        if os.path.exists(self.test_storage_path):
            logger.info(f"Cleaning existing test storage: {self.test_storage_path}")
            shutil.rmtree(self.test_storage_path)
        os.makedirs(self.test_storage_path)
        logger.info(f"Created clean test storage directory: {self.test_storage_path}")
        
        # Initialize components
        self.rag_pipeline = RAGPipeline()
        self.index_manager = IndexManager(self.test_storage_path)
        self.embedding_manager = EmbeddingManager()
        self.document_processor = DocumentProcessor()
        self.embed_model = self.embedding_manager.load_embedding_model()
        
        logger.info("Setup completed. All managers initialized.")
        
    def test_full_workflow(self):
        """
        Test complete system workflow integration.
        
        This test verifies the entire RAG system workflow:
            1. Document Creation and Processing
            2. Embedding Generation and Storage
            3. Index Creation and Management
            4. Query Processing and Response
            
        Test Flow:
            1. Create test documents
            2. Process and embed documents
            3. Create searchable index
            4. Initialize RAG pipeline
            5. Process test queries
            
        Validation Points:
            - Document processing accuracy
            - Embedding quality
            - Index functionality
            - Query response relevance
            
        Note:
            This is a comprehensive test that exercises all major
            system components in sequence.
        """
        # Step 1: Create sample documents with specific test content
        raw_texts = [
            "Python is a high-level programming language known for its simplicity",
            "Python is widely used in data science and machine learning",
            "Software testing is essential for quality assurance"
        ]
        logger.info(f"Starting with {len(raw_texts)} raw texts")
        for i, text in enumerate(raw_texts):
            logger.info(f"Text {i + 1}: {text[:50]}...")
        
        # Step 2: Convert raw texts to Document objects
        documents = self.document_processor.create_documents(raw_texts)
        logger.info(f"Created {len(documents)} Document objects")
        
        # Step 3: Add vector embeddings to documents
        documents = self.document_processor.batch_add_embeddings(self.embed_model, documents)
        logger.info("Added embeddings to documents")
        for i, doc in enumerate(documents):
            logger.info(f"Document {i + 1} has embeddings: {doc.embedding is not None}")
        
        # Step 4: Create a searchable index from the documents
        logger.info("Creating searchable index...")
        index = self.index_manager.create_index(documents, self.embed_model)
        self.assertIsNotNone(index, "Index should be created successfully")
        logger.info("Index created successfully")
        
        # Step 5: Save the index for future use
        logger.info("Saving index to disk...")
        self.index_manager.save_index()
        logger.info(f"Index saved to: {self.index_manager.get_storage_path()}")
        
        # Step 6: Set up the RAG pipeline for querying
        logger.info("Preparing RAG pipeline...")
        self.rag_pipeline.initialize_pipeline(index)
        logger.info("RAG pipeline ready for queries")
        
        # Step 7: Test the system with multiple queries
        test_queries = [
            "What is Python used for?",
            "Why is testing important?"
        ]
        
        for query in test_queries:
            logger.info(f"\nTesting query: {query}")
            results = self.rag_pipeline.run_pipeline(query)
            logger.info(f"Retrieved {len(results)} results")
            for i, result in enumerate(results):
                logger.info(f"Result {i + 1}: {result[:100]}...")
            
            self.assertTrue(len(results) > 0, f"Should get results for query: {query}")
            
            # Verify results are relevant to the query
            if "Python" in query:
                self.assertTrue(
                    any("Python" in result for result in results),
                    "Results should contain information about Python"
                )
            if "testing" in query.lower():
                self.assertTrue(
                    any("testing" in result.lower() for result in results),
                    "Results should contain information about testing"
                )
        
        logger.info("All queries tested successfully")

    def tearDown(self):
        """
        Cleanup method that runs after each test.
        
        Performs:
            1. Pipeline shutdown
            2. Index cleanup
            3. Storage directory removal
            4. Resource deallocation
            
        Note:
            Ensures complete cleanup of all test artifacts and
            resources to prevent test interference.
        """
        logger.info("Starting cleanup process...")
        
        # Clear RAG pipeline
        if hasattr(self.rag_pipeline, 'retriever'):
            self.rag_pipeline.retriever = None
            logger.info("Cleared RAG pipeline retriever")
        
        # Clear index manager
        if hasattr(self.index_manager, 'index'):
            self.index_manager.index = None
            logger.info("Cleared index manager")
        
        # Remove test storage directory
        if os.path.exists(self.test_storage_path):
            try:
                # List contents before removal (for debugging)
                for root, dirs, files in os.walk(self.test_storage_path):
                    logger.info(f"Cleaning up directory: {root}")
                    for f in files:
                        logger.info(f"Removing file: {f}")
                
                # Remove the directory and all its contents
                shutil.rmtree(self.test_storage_path)
                logger.info(f"Successfully removed test storage directory: {self.test_storage_path}")
            except Exception as e:
                logger.error(f"Error during cleanup of {self.test_storage_path}: {str(e)}")
        
        # Clear any cached embeddings
        if hasattr(self.embedding_manager, 'embedding_model'):
            self.embedding_manager.embedding_model = None
            logger.info("Cleared embedding model cache")
        
        logger.info("Cleanup completed successfully")

if __name__ == '__main__':
    unittest.main() 