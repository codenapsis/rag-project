from typing import List
from llama_index.core import Document
from llama_index.readers.file import PDFReader
from llama_index.readers.web import TrafilaturaWebReader
import logging
from ..utils.error_handler import handle_exceptions, DocumentProcessingError

logger = logging.getLogger(__name__)

class ContentHandler:
    """
    ContentHandler is a class that helps you extract text from different sources like 
    PDF files and web pages.

    Think of it as a helper that can:
    - Read and extract text from PDF files
    - Get content from web pages
    - Convert the content into a format that our system can understand

    Example Usage:
        # To read a PDF file:
        documents = ContentHandler.get_data_from_pdf("path/to/file.pdf")

        # To get content from a website:
        documents = ContentHandler.get_data_from_web(["https://example.com"])
    """

    def __init__(self):
        """
        Initializes the DocumentProcessor with an optional embedding model.

        :param embedding_model: An optional embedding model for generating embeddings.
        """

    @handle_exceptions(
        error_message="Failed to extract data from PDF",
        raise_exception=DocumentProcessingError
    )
    def get_data_from_pdf(self, pdf_path: str) -> List[Document]:
        """
        Extracts text from a PDF file and converts it into processable documents.

        How it works:
        1. Takes a PDF file path as input
        2. Reads the PDF file
        3. Extracts all the text content
        4. Converts the text into Document objects that our system can use

        Args:
            pdf_path (str): The path to your PDF file
                Example: "documents/my_file.pdf"

        Returns:
            List[Document]: A list of Document objects containing the extracted text
                Note: Each Document object represents a chunk of text from your PDF

        Raises:
            FileNotFoundError: If the PDF file doesn't exist at the given path
            ValueError: If the PDF is empty or can't be read
            Exception: For any other unexpected errors while processing the PDF

        Example:
            try:
                documents = ContentHandler.get_data_from_pdf("my_file.pdf")
                print(f"Successfully extracted {len(documents)} text chunks")
            except FileNotFoundError:
                print("Oops! The PDF file wasn't found")
        """
        if not pdf_path or not isinstance(pdf_path, str):
            raise ValueError("Please provide a valid file path")

        pdf_reader = PDFReader()
        documents = pdf_reader.load_data(file=pdf_path)

        if not documents:
            raise ValueError("No text could be extracted from the PDF")

        return documents

    @handle_exceptions(
        error_message="Failed to extract data from web",
        raise_exception=DocumentProcessingError
    )
    def get_data_from_web(self, urls: List[str]) -> List[Document]:
        """
        Fetches and processes content from web pages.

        How it works:
        1. Takes a list of URLs as input
        2. Visits each website
        3. Extracts the main content (ignoring ads, menus, etc.)
        4. Converts the content into Document objects

        Args:
            urls (List[str]): A list of website URLs you want to process
                Example: ["https://example.com", "https://another-site.com"]

        Returns:
            List[Document]: A list of Document objects containing the web content
                Note: Each Document contains the main text from a webpage

        Raises:
            ValueError: If the URLs are invalid or no content could be extracted
            Exception: For any errors during web content processing

        Example:
            try:
                urls = ["https://example.com"]
                documents = ContentHandler.get_data_from_web(urls)
                print(f"Successfully extracted content from {len(documents)} pages")
            except Exception as e:
                print(f"Error getting web content: {e}")
        """
        if not urls or not isinstance(urls, list):
            raise ValueError("Please provide a valid list of URLs")

        web_reader = TrafilaturaWebReader()
        documents = web_reader.load_data(urls=urls)

        if not documents:
            raise ValueError("No content could be extracted from the given URLs")

        return documents