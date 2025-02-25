import unittest
import logging
from src.documents.document_processor import DocumentProcessor
from llama_index.core import Document
from src.embeddings.embedding_manager import EmbeddingManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestDocumentProcessor(unittest.TestCase):
    def setUp(self):
        logger.info("Setting up DocumentProcessor test...")
        self.document_processor = DocumentProcessor()
        self.embedding_manager = EmbeddingManager()
        self.embed_model = self.embedding_manager.load_embedding_model()
        logger.info("Setup completed. DocumentProcessor and EmbeddingManager initialized.")
        
    def test_create_documents(self):
        logger.info("Testing create_documents method...")
        raw_texts = ["Test document 1", "Test document 2"]
        logger.info(f"Raw texts to process: {raw_texts}")
        
        documents = self.document_processor.create_documents(raw_texts)
        logger.info(f"Created {len(documents)} documents")
        
        self.assertEqual(len(documents), 2)
        self.assertIsInstance(documents[0], Document)
        logger.info("Document creation test passed")
        
    def test_add_embeddings(self):
        logger.info("Testing add_embeddings method...")
        document = Document(text="Test document")
        logger.info(f"Created test document with text: {document.text}")
        
        processed_doc = self.document_processor.add_embeddings(self.embed_model, document)
        logger.info(f"Added embeddings. Embedding size: {len(processed_doc.embedding) if processed_doc.embedding else 'No embedding'}")
        
        self.assertIsNotNone(processed_doc.embedding)
        logger.info("Embedding addition test passed")

    def tearDown(self):
        logger.info("Cleaning up after DocumentProcessor test...") 