"""
Test suite for ContentHandler component.

This test suite verifies the functionality of the ContentHandler class, which is responsible
for extracting and processing content from different sources (PDF files and web pages).

Key Components Tested:
    - PDF data extraction functionality
    - Web content extraction functionality
    - Document creation from extracted content
    
Test Environment:
    - Uses real file system for PDF testing
    - Makes real HTTP requests for web content testing
    - Requires test PDF files in the 'pdfs' directory
    
Dependencies:
    - BaseTest: Provides common test setup and teardown functionality
    - ContentHandler: Main component being tested
    - File system access for PDF operations
    - Network access for web content retrieval

Note:
    These tests interact with external resources (files and web) and may need
    special consideration in CI/CD environments or offline testing scenarios.
"""

import unittest
import logging
import os
from tests.base_test import BaseTest
from src.content.content_handler import ContentHandler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestContentHandler(BaseTest):
    """
    Test cases for ContentHandler functionality.
    
    This test class verifies the ability to extract and process content from
    different sources. It tests both PDF and web content extraction, ensuring
    that the ContentHandler can properly handle different input types and
    create appropriate document objects.
    
    This test class inherits from BaseTest which provides:
        - Common test file paths
        - Standard test URLs
        - Logging configuration
        
    Test Prerequisites:
        - PDF files must exist in the specified test directory
        - Network access must be available for web content tests
        - Appropriate permissions for file and network operations
    """
    
    def setUp(self):
        """
        Test setup method that runs before each test.
        
        Sets up:
            1. Base test configuration through parent class
            2. ContentHandler instance for testing
            
        Note:
            Unlike other components, ContentHandler doesn't require
            special mocking as it directly interfaces with files and web.
        """
        super().setUp()
        self.content_handler = ContentHandler()
        logger.info("Setup completed. ContentHandler initialized.")
        
    def test_get_data_from_pdf(self):
        """
        Test PDF content extraction functionality.
        
        This test verifies that:
            1. PDF files can be located and accessed
            2. Content can be extracted from PDFs
            3. Extracted content is converted to documents
            4. Documents contain valid text content
            
        Test Flow:
            1. Check for test PDF existence
            2. Extract content from PDF
            3. Verify document creation
            4. Validate document content
            
        Error Handling:
            - Skips test if PDF file is not found
            - Logs warning messages for missing files
            
        Note:
            This test requires actual PDF files and might need modification
            in environments where test files cannot be included.
        """
        logger.info("Testing PDF data extraction...")
        if not os.path.exists(self.TEST_PDF_PATH):
            logger.warning(f"Test PDF file not found at {self.TEST_PDF_PATH}")
            self.skipTest(f"Test PDF file not found at {self.TEST_PDF_PATH}")
        
        documents = self.content_handler.get_data_from_pdf(self.TEST_PDF_PATH)
        logger.info(f"Extracted {len(documents)} documents from PDF")
        for i, doc in enumerate(documents):
            logger.info(f"Document {i} length: {len(doc.text)} characters")
        
        self.assertTrue(len(documents) > 0)
        logger.info("PDF extraction test passed")
        
    def test_get_data_from_web(self):
        """
        Test web content extraction functionality.
        
        This test verifies that:
            1. Web content can be retrieved from URLs
            2. Retrieved content is properly processed
            3. Content is converted to document format
            
        Test Flow:
            1. Attempt to retrieve content from test URLs
            2. Process retrieved content
            3. Verify document creation
            4. Validate document content
            
        Dependencies:
            - Network access
            - Target website availability
            
        Note:
            This test makes actual HTTP requests and may fail due to:
            - Network connectivity issues
            - Target website unavailability
            - Rate limiting
            - Changed website content
            
        Consider implementing mock responses for more reliable testing
        in CI/CD environments.
        """
        logger.info("Testing web content extraction...")
        documents = self.content_handler.get_data_from_web(self.TEST_URLS)
        logger.info(f"Extracted {len(documents)} documents from web")
        for i, doc in enumerate(documents):
            logger.info(f"Document {i} length: {len(doc.text)} characters")
        
        self.assertTrue(len(documents) > 0)
        logger.info("Web extraction test passed")

    def tearDown(self):
        logger.info("Cleaning up after ContentHandler test...") 