from llama_index.core import VectorStoreIndex, Document, StorageContext, load_index_from_storage
from llama_index.core.indices.base import BaseIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.storage.storage_context import StorageContext
from typing import List, Optional
import os
import logging
import hashlib
from src.utils.error_handler import handle_exceptions, IndexError

logger = logging.getLogger(__name__)

class IndexManager:
    """
    IndexManager handles the creation, storage, and retrieval of searchable document indexes.
    
    What is an Index?
    - Think of it as a smart library catalog for your documents
    - It organizes documents so you can quickly find relevant information
    - Uses AI to understand and match document content
    
    Key Features:
    1. Creates searchable indexes from documents
    2. Saves indexes to disk (so you don't need to rebuild them)
    3. Loads existing indexes from disk
    4. Helps find relevant information in your documents

    Example Usage:
        # Create a new index
        manager = IndexManager("my_index_folder")
        index = manager.create_index(documents, embedding_model)
        
        # Save it for later
        manager.save_index()
        
        # Load it back when needed
        loaded_index = manager.load_index(embedding_model)
        
        # Search for information
        result = manager.query_index("What is Python?")
    """

    def __init__(self, storage_path: str = "index_storage"):
        """
        Initializes the IndexManager with a specific storage location.

        Args:
            storage_path (str): Where to save/load the index files
                Default is "index_storage"
                Example: "my_project/indexes"

        Note:
            - Creates the storage directory if it doesn't exist
            - Each index needs its own storage path
        """
        self.storage_path = storage_path
        os.makedirs(self.storage_path, exist_ok=True)
        self.index = None
        logger.info(f"Initialized IndexManager with storage path: {storage_path}")

    def _ensure_document_ids(self, documents: List[Document]) -> List[Document]:
        """
        Ensures each document has a unique ID using the id_ property.
        
        :param documents: List of documents to process
        :return: List of documents with assigned IDs
        """
        for doc in documents:
            if not doc.id_:
                # Create a unique hash based on document content
                content_hash = hashlib.sha256(doc.text.encode()).hexdigest()[:16]
                doc.id_ = f"doc_{content_hash}"
                logger.debug(f"Assigned ID {doc.id_} to document")
        return documents

    @handle_exceptions(
        error_message="Failed to create index",
        raise_exception=IndexError
    )
    def create_index(self, data: List[Document], embed_model: BaseEmbedding) -> VectorStoreIndex:
        """
        Creates a searchable index from a list of documents.

        How it works:
        1. Takes your documents and an embedding model
        2. Processes each document to understand its content
        3. Creates a searchable index structure
        4. Stores everything in memory (use save_index() to save to disk)

        Args:
            data (List[Document]): Your documents to index
                These should be Document objects (created by DocumentProcessor)
            embed_model (BaseEmbedding): The model to understand document content
                This should come from EmbeddingManager

        Returns:
            VectorStoreIndex: A searchable index of your documents

        Example:
            # First, get your documents and embedding model ready
            documents = document_processor.create_documents(texts)
            embed_model = embedding_manager.load_embedding_model()
            
            # Then create the index
            index = manager.create_index(documents, embed_model)
        """
        logger.info(f"Creating index with {len(data)} documents...")
        
        # Ensure all documents have proper IDs
        data = self._ensure_document_ids(data)
        
        # Create storage context
        storage_context = StorageContext.from_defaults()
        
        # Create the index
        self.index = VectorStoreIndex.from_documents(
            documents=data,
            storage_context=storage_context,
            embed_model=embed_model,
            store_nodes_override=True,
            show_progress=False
        )
        
        logger.info("Index created successfully")
        return self.index

    @handle_exceptions(
        error_message="Failed to save index",
        raise_exception=IndexError
    )
    def save_index(self) -> None:
        """
        Saves the current index to disk so you can load it later.

        Why save the index?
        - Saves time (don't need to rebuild the index)
        - Preserves your work
        - Can share the index with others

        Raises:
            ValueError: If there's no index to save

        Example:
            manager = IndexManager("my_indexes")
            manager.create_index(documents, embed_model)
            manager.save_index()  # Save for later use
        """
        if not self.index:
            raise ValueError("No index to save. Create an index first.")

        self.index.storage_context.persist(persist_dir=self.storage_path)
        doc_count = len(self.index.docstore.docs)
        logger.info(f"Index saved to {self.storage_path} with {doc_count} documents")

    @handle_exceptions(
        error_message="Failed to load index",
        raise_exception=IndexError
    )
    def load_index(self, embed_model: BaseEmbedding) -> BaseIndex:
        """
        Loads a previously saved index from disk.

        Why load an index?
        - Faster than creating a new one
        - Continues work from a previous session
        - Shares indexes between different parts of your program

        Args:
            embed_model (BaseEmbedding): The embedding model to use with the index
                Must be the same type used when creating the index

        Returns:
            BaseIndex: The loaded index, ready for searching

        Raises:
            FileNotFoundError: If the index files don't exist
            ValueError: If there's a problem loading the index

        Example:
            manager = IndexManager("my_indexes")
            embed_model = embedding_manager.load_embedding_model()
            index = manager.load_index(embed_model)  # Load existing index
        """
        if not os.path.exists(self.storage_path):
            raise FileNotFoundError(f"Index storage path {self.storage_path} does not exist")
        
        storage_context = StorageContext.from_defaults(persist_dir=self.storage_path)
        self.index = load_index_from_storage(
            storage_context,
            embed_model=embed_model,
            store_nodes_override=True,
            show_progress=False
        )
        doc_count = len(self.index.docstore.docs)
        logger.info(f"Index loaded from {self.storage_path} with {doc_count} documents")
        return self.index

    def query_index(self, query: str) -> str:
        """
        Searches the index for information related to your query.

        How it works:
        1. Takes your question or query
        2. Searches through the indexed documents
        3. Returns the most relevant information

        Args:
            query (str): Your question or search term
                Example: "What is machine learning?"

        Returns:
            str: The answer or relevant information found

        Raises:
            ValueError: If there's no index to search

        Example:
            manager = IndexManager("my_indexes")
            # ... load or create index ...
            result = manager.query_index("What is Python used for?")
            print(result)
        """
        if not self.index:
            raise ValueError("No index loaded. Create or load an index first.")

        query_engine = self.index.as_query_engine()
        response = query_engine.query(query)
        return str(response)

    def get_storage_path(self) -> str:
        """
        Gets the path where this index is stored.

        Returns:
            str: The storage path for this index

        Example:
            manager = IndexManager("my_indexes")
            path = manager.get_storage_path()
            print(f"Index is stored in: {path}")
        """
        return self.storage_path