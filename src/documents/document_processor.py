from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document
from typing import List, Optional
import logging
from src.utils.error_handler import handle_exceptions, DocumentProcessingError

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    DocumentProcessor is a class that helps prepare text documents for our AI system.
    It handles two main tasks:
    1. Converting raw text into Document objects
    2. Adding embeddings (numerical representations) to these documents

    Think of it as a document preparation service that gets your text ready for AI processing.

    Key Concepts:
    - Document: A special container for text that our AI system can understand
    - Embedding: A way to convert text into numbers that capture its meaning
    - Batch Processing: Processing multiple documents at once for efficiency

    Example Usage:
        processor = DocumentProcessor()
        
        # Convert texts to documents
        texts = ["Hello world", "Another text"]
        documents = processor.create_documents(texts)
        
        # Add embeddings to documents
        embedding_model = some_embedding_model
        documents_with_embeddings = processor.batch_add_embeddings(embedding_model, documents)
    """

    def __init__(self):
        """
        Initializes a new DocumentProcessor.
        Currently, this is empty but allows for future additions of initialization parameters.
        """
        pass

    @handle_exceptions(
        error_message="Failed to create documents",
        raise_exception=DocumentProcessingError
    )
    def create_documents(self, raw_texts: List[str]) -> List[Document]:
        """
        Converts a list of raw text strings into Document objects that our AI system can process.

        How it works:
        1. Takes a list of text strings
        2. Converts each text into a Document object
        3. Returns a list of these Document objects

        Args:
            raw_texts (List[str]): A list of text strings you want to process
                Example: ["First text", "Second text", "Third text"]

        Returns:
            List[Document]: A list of Document objects ready for further processing
                Note: These documents don't have embeddings yet

        Example:
            processor = DocumentProcessor()
            texts = [
                "Python is a programming language",
                "Machine learning is fascinating"
            ]
            documents = processor.create_documents(texts)
            print(f"Created {len(documents)} documents")
        """
        documents = []
        for text in raw_texts:
            document = Document(text=text)
            documents.append(document)
            
        return documents

    @handle_exceptions(
        error_message="Failed to add embeddings",
        raise_exception=DocumentProcessingError
    )
    def add_embeddings(self, embedding_model, document: Document) -> Document:
        """
        Adds numerical embeddings to a single document.

        How it works:
        1. Takes a document and an embedding model
        2. Converts the document's text into a numerical representation (embedding)
        3. Attaches this embedding to the document

        Args:
            embedding_model: The model to use for creating embeddings
                Note: This is usually a pre-trained AI model
            document (Document): The document to process

        Returns:
            Document: The same document, but now with embeddings attached

        Raises:
            ValueError: If the embedding model isn't properly configured

        Example:
            processor = DocumentProcessor()
            doc = Document(text="Hello world")
            doc_with_embedding = processor.add_embeddings(embedding_model, doc)
        """
        if not embedding_model:
            raise ValueError("Embedding model is not configured")
        
        embedding = embedding_model.get_text_embedding(document.text)
        document.embedding = embedding
        return document

    @handle_exceptions(
        error_message="Failed to add batch embeddings",
        raise_exception=DocumentProcessingError
    )
    def batch_add_embeddings(self, embedding_model, documents: List[Document]) -> List[Document]:
        """
        Adds embeddings to multiple documents at once (batch processing).

        How it works:
        1. Takes a list of documents and an embedding model
        2. Processes each document to add embeddings
        3. Returns all documents with their embeddings

        Why use this?
        - It's more efficient than processing documents one by one
        - Keeps your code cleaner when working with multiple documents

        Args:
            embedding_model: The model to use for creating embeddings
            documents (List[Document]): List of documents to process

        Returns:
            List[Document]: The same documents, but now with embeddings

        Example:
            processor = DocumentProcessor()
            docs = [
                Document(text="First document"),
                Document(text="Second document")
            ]
            docs_with_embeddings = processor.batch_add_embeddings(embedding_model, docs)
            print(f"Processed {len(docs_with_embeddings)} documents")
        """
        return [self.add_embeddings(embedding_model, doc) for doc in documents]

    def extract_metadata(self, document: Document) -> Document:
        document.extra_info = {"title": document.text.split(".")[0]}  # First sentence as title
        return document

    def preprocess_text(self, document: Document) -> Document:
        document.set_content(document.text.strip().lower())
        return document
