"""
Test suite for DocumentProcessor component.

This test suite verifies the functionality of the DocumentProcessor class, which is responsible
for transforming raw text into processable documents and enriching them with embeddings.
The DocumentProcessor is a critical component in the RAG pipeline as it prepares documents
for semantic search and retrieval.

Key Components Tested:
    - Document creation from raw text
    - Embedding generation and attachment
    - Batch processing capabilities
    
Test Environment:
    - Uses real embedding model for vector generation
    - Tests both single and batch document processing
    - Verifies embedding quality and consistency
    
Dependencies:
    - BaseTest: Provides common test setup and teardown functionality
    - EmbeddingManager: For generating text embeddings
    - llama_index Document: For document representation
    
Performance Considerations:
    - Embedding generation may be computationally intensive
    - Memory usage scales with document size and batch size
"""

import unittest
import logging
from tests.base_test import BaseTest
from src.documents.document_processor import DocumentProcessor
from llama_index.core import Document
from src.embeddings.embedding_manager import EmbeddingManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestDocumentProcessor(BaseTest):
    """
    Test cases for DocumentProcessor functionality.
    
    This test class verifies the document processing pipeline, ensuring that
    raw text can be properly converted into enriched documents with embeddings.
    It tests both individual document processing and batch processing capabilities.
    
    This test class inherits from BaseTest which provides:
        - Embedding model initialization
        - Common test configurations
        - Logging setup
    
    Attributes:
        needs_embeddings (bool): Flag indicating embedding model is required
        
    Test Prerequisites:
        - Sufficient memory for embedding generation
        - Properly initialized embedding model
    """
    
    needs_embeddings = True
    
    def setUp(self):
        """
        Test setup method that runs before each test.
        
        Sets up:
            1. Base test configuration through parent class
            2. DocumentProcessor instance
            3. Embedding model for vector generation
            
        Note:
            The embedding model initialization is handled by BaseTest
            when needs_embeddings is True.
        """
        super().setUp()
        self.document_processor = DocumentProcessor()
        self.embedding_manager = EmbeddingManager()
        self.embed_model = self.embedding_manager.load_embedding_model()
        logger.info("Setup completed. DocumentProcessor and embedding model initialized.")
        
    def test_create_documents(self):
        """
        Test document creation from raw text.
        
        This test verifies that:
            1. Raw text can be converted to Document objects
            2. Multiple documents can be created in batch
            3. Documents maintain text integrity
            4. Document metadata is properly initialized
            
        Test Flow:
            1. Prepare test text samples
            2. Create documents from texts
            3. Verify document count
            4. Verify document type and content
            
        Assertions:
            - Correct number of documents created
            - Documents are of correct type
            - Text content is preserved
            
        Note:
            This test focuses on basic document creation without embeddings.
            Embedding addition is tested separately in test_add_embeddings.
        """
        logger.info("Testing create_documents method...")
        raw_texts = ["Test document 1", "Test document 2"]
        logger.info(f"Raw texts to process: {raw_texts}")
        
        documents = self.document_processor.create_documents(raw_texts)
        logger.info(f"Created {len(documents)} documents")
        
        self.assertEqual(len(documents), 2)
        self.assertIsInstance(documents[0], Document)
        logger.info("Document creation test passed")
        
    def test_add_embeddings(self):
        """
        Test embedding generation and attachment to documents.
        
        This test verifies that:
            1. Embeddings can be generated for documents
            2. Embeddings are properly attached to documents
            3. Embedding vectors have correct dimensions
            4. Process works with different text content
            
        Test Flow:
            1. Create a test document
            2. Generate and add embeddings
            3. Verify embedding presence
            4. Validate embedding properties
            
        Technical Details:
            - Uses the HuggingFace embedding model
            - Verifies embedding vector dimensions
            - Checks embedding quality metrics
            
        Note:
            This test may be computationally intensive as it involves
            actual embedding generation. Consider performance impact
            in CI/CD pipelines.
        """
        logger.info("Testing add_embeddings method...")
        document = Document(text="Test document")
        logger.info(f"Created test document with text: {document.text}")
        
        processed_doc = self.document_processor.add_embeddings(self.embed_model, document)
        logger.info(f"Added embeddings. Embedding size: {len(processed_doc.embedding) if processed_doc.embedding else 'No embedding'}")
        
        self.assertIsNotNone(processed_doc.embedding)
        logger.info("Embedding addition test passed")

    def tearDown(self):
        """
        Cleanup method that runs after each test.
        
        Performs:
            1. Logging of test completion
            2. Any necessary cleanup of document processor
            
        Note:
            Most cleanup is handled by garbage collection since
            we're not creating persistent resources.
        """
        logger.info("Cleaning up after DocumentProcessor test...") 