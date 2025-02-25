import unittest
from src.embeddings.embedding_manager import EmbeddingManager
from llama_index.core import Settings

class TestEmbeddingManager(unittest.TestCase):
    def setUp(self):
        Settings.llm = None
        self.embedding_manager = EmbeddingManager()
        
    def test_load_embedding_model(self):
        embed_model = self.embedding_manager.load_embedding_model()
        self.assertIsNotNone(embed_model)
        
    def test_get_embeddings(self):
        texts = ["Test text"]
        embeddings = self.embedding_manager.get_embeddings(texts)
        self.assertIsNotNone(embeddings)
        self.assertTrue(len(embeddings) > 0) 