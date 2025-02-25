import unittest
from src.ingestion.data_ingestion_manager import DataIngestionManager
from llama_index.core import Document, Settings

class TestDataIngestionManager(unittest.TestCase):
    def setUp(self):
        Settings.llm = None
        self.data_ingestion_manager = DataIngestionManager()
        
    def test_configure_pipeline(self):
        pipeline = self.data_ingestion_manager.configure_pipeline(
            docstore=None,
            embedding_model=self.embed_model
        )
        self.assertIsNotNone(pipeline)
        
    def test_ingest_from_documents(self):
        documents = [
            Document(text="Test document")
        ]
        pipeline = self.data_ingestion_manager.configure_pipeline(
            docstore=None,
            embedding_model=self.embed_model
        )
        self.data_ingestion_manager.ingest_from_documents(documents, pipeline)
