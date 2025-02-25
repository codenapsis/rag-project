import unittest
from src.index.index_manager import IndexManager
from src.embeddings.embedding_manager import EmbeddingManager
from llama_index.core import Document, Settings
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever

class TestIndexManager(unittest.TestCase):
    def setUp(self):
        # Disable LLM for tests
        Settings.llm = None
        Settings.context_window = 2048
        Settings.num_output = 256
        
        # Create an IndexManager instance for each test
        self.index_manager = IndexManager("test_storage")
        # Use our EmbeddingManager
        self.embedding_manager = EmbeddingManager()
        self.embed_model = self.embedding_manager.load_embedding_model()
        
    def test_create_and_save_index(self):
        # Test index creation and saving
        documents = [Document(text="Test document")]
        index = self.index_manager.create_index(documents, self.embed_model)
        self.assertIsNotNone(index)
        self.index_manager.save_index()
        
    def test_load_index(self):
        # Test index loading
        loaded_index = self.index_manager.load_index(self.embed_model)
        self.assertIsNotNone(loaded_index)

if __name__ == '__main__':
    unittest.main()
