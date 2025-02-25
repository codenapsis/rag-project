import unittest
import logging
from src.pipeline.rag_pipeline import RAGPipeline
from src.index.index_manager import IndexManager
from src.embeddings.embedding_manager import EmbeddingManager
from llama_index.core import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestRAGPipeline(unittest.TestCase):
    def setUp(self):
        logger.info("Setting up RAGPipeline test...")
        self.rag_pipeline = RAGPipeline()
        self.index_manager = IndexManager("test_storage")
        self.embedding_manager = EmbeddingManager()
        self.embed_model = self.embedding_manager.load_embedding_model()
        logger.info("Setup completed. All components initialized.")
        
    def test_initialize_pipeline(self):
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
        logger.info("Cleaning up after RAGPipeline test...")
