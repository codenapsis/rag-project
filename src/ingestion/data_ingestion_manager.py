from typing import List, Optional
from llama_index.core import Document
from src.content.content_handler import ContentHandler
from src.documents.document_processor import DocumentProcessor
from src.embeddings.embedding_manager import EmbeddingManager
from src.index.index_manager import IndexManager
import logging

logger = logging.getLogger(__name__)

class DataIngestionManager:
    """
    DataIngestionManager is like a master chef in a kitchen - it coordinates all the steps
    needed to prepare your data for AI processing.

    What does it do?
    1. Gets raw content (like text from PDFs or websites)
    2. Processes this content into a format the AI can understand
    3. Adds special AI-friendly numbers (embeddings) to the content
    4. Creates a searchable index of all the information

    Think of it as a pipeline:
    Raw Content → Processed Documents → Documents with Embeddings → Searchable Index

    Example Usage:
        # Initialize the manager
        manager = DataIngestionManager()

        # Process a PDF file
        manager.ingest_pdf("my_document.pdf")

        # Or process some websites
        urls = ["https://example.com", "https://another-site.com"]
        manager.ingest_web_content(urls)
    """

    def __init__(self, storage_path: str = "index_storage"):
        """
        Sets up all the tools needed for data processing.

        Args:
            storage_path (str): Where to save the processed data
                Default: "index_storage"

        Think of this as setting up your workstation with all the tools you'll need:
        - ContentHandler: Gets raw content from sources
        - DocumentProcessor: Prepares documents for AI processing
        - EmbeddingManager: Adds AI-understanding capabilities
        - IndexManager: Organizes everything for quick searching
        """
        self.content_handler = ContentHandler()
        self.document_processor = DocumentProcessor()
        self.embedding_manager = EmbeddingManager()
        self.index_manager = IndexManager(storage_path)
        logger.info("DataIngestionManager initialized with all components")

    def ingest_pdf(self, pdf_path: str) -> None:
        """
        Processes a PDF file from start to finish.

        How it works:
        1. Reads the PDF file
        2. Converts PDF content into processable documents
        3. Adds AI embeddings to understand the content
        4. Creates a searchable index

        Args:
            pdf_path (str): Path to your PDF file
                Example: "documents/my_book.pdf"

        Example:
            manager = DataIngestionManager()
            try:
                manager.ingest_pdf("my_textbook.pdf")
                print("PDF processed successfully!")
            except Exception as e:
                print(f"Oops! Something went wrong: {e}")
        """
        try:
            # Step 1: Extract content from PDF
            logger.info(f"Starting PDF ingestion: {pdf_path}")
            raw_documents = self.content_handler.get_data_from_pdf(pdf_path)
            logger.info(f"Extracted {len(raw_documents)} documents from PDF")

            # Step 2: Process the documents
            self._process_documents(raw_documents)
            logger.info("PDF processing completed successfully")

        except Exception as e:
            logger.error(f"Error during PDF ingestion: {str(e)}")
            raise

    def ingest_web_content(self, urls: List[str]) -> None:
        """
        Processes content from websites.

        How it works:
        1. Visits each website and extracts the main content
        2. Converts web content into processable documents
        3. Adds AI embeddings to understand the content
        4. Creates a searchable index

        Args:
            urls (List[str]): List of website URLs to process
                Example: ["https://example.com", "https://another-site.com"]

        Example:
            manager = DataIngestionManager()
            websites = [
                "https://python.org",
                "https://pytorch.org"
            ]
            try:
                manager.ingest_web_content(websites)
                print("Websites processed successfully!")
            except Exception as e:
                print(f"Oops! Something went wrong: {e}")
        """
        try:
            # Step 1: Extract content from websites
            logger.info(f"Starting web content ingestion for {len(urls)} URLs")
            raw_documents = self.content_handler.get_data_from_web(urls)
            logger.info(f"Extracted {len(raw_documents)} documents from web")

            # Step 2: Process the documents
            self._process_documents(raw_documents)
            logger.info("Web content processing completed successfully")

        except Exception as e:
            logger.error(f"Error during web content ingestion: {str(e)}")
            raise

    def _process_documents(self, raw_documents: List[Document]) -> None:
        """
        Internal helper method that processes documents through the AI pipeline.

        How it works:
        1. Prepares documents for AI processing
        2. Adds embeddings (AI understanding)
        3. Creates a searchable index
        4. Saves everything for later use

        Args:
            raw_documents (List[Document]): Documents to process

        Note:
            This is an internal method (that's why it starts with _)
            It's used by both ingest_pdf and ingest_web_content
        """
        try:
            # Step 1: Load the embedding model
            logger.info("Loading embedding model...")
            embed_model = self.embedding_manager.load_embedding_model()

            # Step 2: Add embeddings to documents
            logger.info("Adding embeddings to documents...")
            processed_docs = self.document_processor.batch_add_embeddings(
                embed_model, raw_documents
            )
            logger.info(f"Added embeddings to {len(processed_docs)} documents")

            # Step 3: Create and save the searchable index
            logger.info("Creating searchable index...")
            self.index_manager.create_index(processed_docs, embed_model)
            self.index_manager.save_index()
            logger.info("Index created and saved successfully")

        except Exception as e:
            logger.error(f"Error during document processing: {str(e)}")
            raise

