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

"""
Integration Tests for the RAG (Retrieval-Augmented Generation) System

This test suite verifies that all components of the system work together correctly.
It tests the complete workflow from document creation to query processing.

Components tested:
- Document Processing
- Embedding Generation
- Index Creation and Storage
- RAG Pipeline Query Processing
"""

# Set up logging to track what's happening during tests
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestIntegration(unittest.TestCase):
    def setUp(self):
        """
        Set up the test environment before each test.
        This method is called before every test method.
        
        Initializes:
        - RAG Pipeline: Manages the retrieval and generation process
        - Index Manager: Handles document indexing and storage
        - Embedding Manager: Creates vector embeddings for documents
        - Document Processor: Processes raw text into document objects
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
        Tests the complete workflow of the RAG system.
        
        Steps tested:
        1. Create documents from raw text
        2. Add embeddings to documents
        3. Create and save an index
        4. Initialize the RAG pipeline
        5. Run a query through the pipeline
        
        This test ensures that all components can work together
        to process documents and answer queries.
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
        Clean up after each test.
        This method is called after every test method.
        
        Cleanup tasks:
        1. Remove the test storage directory and all its contents
        2. Clear any cached data in the managers
        3. Log all cleanup operations
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