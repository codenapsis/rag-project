import unittest
import logging
import os
from src.content.content_handler import ContentHandler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestContentHandler(unittest.TestCase):
    def setUp(self):
        logger.info("Setting up ContentHandler test...")
        self.test_pdf_path = os.path.join(os.getcwd(), "pdfs", "carrotbc.pdf")
        self.test_urls = ["http://example.com"]
        logger.info(f"Test PDF path: {self.test_pdf_path}")
        logger.info(f"Test URLs: {self.test_urls}")
        
    def test_get_data_from_pdf(self):
        logger.info("Testing PDF data extraction...")
        if not os.path.exists(self.test_pdf_path):
            logger.warning(f"Test PDF file not found at {self.test_pdf_path}")
            self.skipTest(f"Test PDF file not found at {self.test_pdf_path}")
        
        documents = ContentHandler.get_data_from_pdf(self.test_pdf_path)
        logger.info(f"Extracted {len(documents)} documents from PDF")
        for i, doc in enumerate(documents):
            logger.info(f"Document {i} length: {len(doc.text)} characters")
        
        self.assertTrue(len(documents) > 0)
        logger.info("PDF extraction test passed")
        
    def test_get_data_from_web(self):
        logger.info("Testing web data extraction...")
        documents = ContentHandler.get_data_from_web(self.test_urls)
        logger.info(f"Extracted {len(documents)} documents from web")
        for i, doc in enumerate(documents):
            logger.info(f"Document {i} length: {len(doc.text)} characters")
        
        self.assertTrue(len(documents) > 0)
        logger.info("Web extraction test passed")

    def tearDown(self):
        logger.info("Cleaning up after ContentHandler test...") 