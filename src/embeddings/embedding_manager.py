from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.embeddings import BaseEmbedding
from typing import Optional
import logging
import os
from src.utils.error_handler import handle_exceptions, EmbeddingError

# Set environment variables to disable progress bars
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Fix tqdm configuration
import tqdm
tqdm.tqdm.disable = True

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """
    EmbeddingManager handles the creation and management of text embeddings.
    
    What are embeddings?
    - Embeddings are numerical representations of text
    - They convert words and sentences into lists of numbers
    - These numbers capture the meaning of the text
    - Similar texts will have similar number patterns
    
    For example:
    - "I love dogs" and "I like puppies" would have similar embeddings
    - "Python programming" and "cooking recipes" would have very different embeddings
    
    This class uses the BAAI/bge-small-en model, which is:
    - Fast and efficient
    - Good at understanding English text
    - Suitable for most basic text processing needs
    
    Example Usage:
        # Create a manager and load the embedding model
        manager = EmbeddingManager()
        model = manager.load_embedding_model()
        
        # Use the model to create embeddings
        text = "Hello, world!"
        embedding = model.get_text_embedding(text)
    """

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """
        Initializes the EmbeddingManager with a specific model.

        Args:
            model_name (str): The name of the embedding model to use
                Default is "sentence-transformers/all-mpnet-base-v2" which is:
                - More accurate for general text understanding
                - Well-suited for document similarity tasks
                - Widely used in production environments
        """
        self.model_name = model_name
        self.embedding_model = None
        logger.info(f"Initialized EmbeddingManager with model: {model_name}")

    @handle_exceptions(
        error_message="Failed to load embedding model",
        raise_exception=EmbeddingError
    )
    def load_embedding_model(self) -> BaseEmbedding:
        """
        Loads and prepares the embedding model for use.

        How it works:
        1. Checks if the model is already loaded (to avoid loading it twice)
        2. If not loaded, downloads and initializes the model
        3. Returns the ready-to-use model

        Returns:
            BaseEmbedding: A model that can convert text to embeddings
                You can use this model with document_processor to add
                embeddings to your documents

        Example:
            manager = EmbeddingManager()
            
            # Load the model
            model = manager.load_embedding_model()
            
            # Now you can use it to create embeddings
            # (though usually document_processor will do this for you)
            embedding = model.get_text_embedding("Some text")

        Note:
            - The first time you run this, it might take a few moments
              as it downloads the model
            - After that, it will be much faster as it uses the cached model
        """
        if self.embedding_model:
            logger.info("Using existing embedding model")
            return self.embedding_model

        logger.info(f"Loading embedding model: {self.model_name}")
        self.embedding_model = HuggingFaceEmbedding(
            model_name=self.model_name
        )
        Settings.embed_model = self.embedding_model
        logger.info("Embedding model loaded successfully")
        return self.embedding_model

    def get_model_name(self) -> str:
        """
        Returns the name of the current embedding model.

        Returns:
            str: The name of the model (e.g., "sentence-transformers/all-mpnet-base-v2")

        Example:
            manager = EmbeddingManager()
            model_name = manager.get_model_name()
            print(f"Using model: {model_name}")
        """
        return self.model_name

    def get_embeddings(self, texts: list[str]) -> list[Optional[list[float]]]:
        """
        Generates embeddings for a list of input texts.

        :param texts: A list of strings to generate embeddings for.
        :return: A list of embedding vectors.
        """
        if not self.embedding_model:
            self.load_embedding_model()

        embeddings = []
        for text in texts:
            try:
                embedding = self.embedding_model.get_text_embedding(text)
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error generating embedding for text: {text[:50]}...: {e}")
                embeddings.append(None)

        return embeddings
