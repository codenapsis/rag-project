"""
Test suite for Error Handler component.

This test suite verifies the functionality of the error handling system in the RAG pipeline.
The error handler provides a robust way to catch, process, and raise appropriate exceptions
throughout the system, ensuring graceful error handling and meaningful error messages.

Key Components Tested:
    - Exception decoration and wrapping
    - Custom exception hierarchy
    - Error message formatting
    - Expected vs unexpected error handling
    
Test Environment:
    - Uses controlled error scenarios
    - Tests both synchronous and decorator-based error handling
    - Verifies error propagation and transformation
    
Dependencies:
    - BaseTest: Provides common test setup and teardown functionality
    - Custom exception classes from error_handler module
    
Design Pattern:
    - Decorator pattern for exception handling
    - Factory pattern for error creation
    - Chain of responsibility for error propagation
"""

import unittest
import logging
from src.utils.error_handler import (
    handle_exceptions,
    RAGError,
    DocumentProcessingError,
    IndexError,
    QueryError,
    EmbeddingError
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestErrorHandler(unittest.TestCase):
    """
    Test cases for error handling functionality.
    
    This test class verifies the complete error handling system, ensuring that
    exceptions are properly caught, transformed, and propagated through the
    system with appropriate context and information.
    
    Key Features Tested:
        - Exception decoration
        - Error transformation
        - Message formatting
        - Error hierarchy
        
    Test Prerequisites:
        - None (self-contained tests)
    """

    def setUp(self):
        """
        Test setup method that runs before each test.
        
        Sets up:
            1. Logging configuration
            2. Error expectation notification
            
        Note:
            These tests intentionally generate errors, so we inform
            about expected error messages in logs.
        """
        logger.info(
            "\n=== NOTE: The following errors are expected and part of the tests ==="
        )

    def test_handle_exceptions_basic(self):
        """
        Test basic error handling functionality.
        
        This test verifies that:
            1. Basic exceptions are caught
            2. Error messages are properly formatted
            3. Exceptions are transformed to RAGError
            
        Test Flow:
            1. Define function that raises error
            2. Decorate with error handler
            3. Execute and verify error transformation
            
        Expected Behavior:
            ValueError should be caught and transformed into RAGError
            with appropriate message formatting.
        """
        logger.info(
            "Testing basic error handling - Expect to see 'Something went wrong' error"
        )
        @handle_exceptions(error_message="Test error")
        def failing_function():
            raise ValueError("Something went wrong")

        with self.assertRaises(RAGError) as context:
            failing_function()
        self.assertIn("Test error", str(context.exception))
        self.assertIn("Something went wrong", str(context.exception))

    def test_handle_exceptions_custom_exception(self):
        """
        Test custom exception handling and transformation.
        
        This test verifies that:
            1. Custom exception types are properly handled
            2. Exceptions are transformed to specified type
            3. Error context is preserved
            
        Test Flow:
            1. Define function with custom error
            2. Specify custom exception transformation
            3. Verify error type and message
        """
        logger.info(
            "Testing custom exception - Expect to see 'Document processing failed' error"
        )
        @handle_exceptions(
            error_message="Document error",
            raise_exception=DocumentProcessingError
        )
        def failing_document_function():
            raise ValueError("Document processing failed")

        with self.assertRaises(DocumentProcessingError) as context:
            failing_document_function()
        self.assertIn("Document error", str(context.exception))

    def test_handle_exceptions_expected_exceptions(self):
        """
        Test handling of specifically expected exceptions.
        
        This test verifies that:
            1. Expected exceptions are caught and transformed
            2. Multiple exception types can be handled
            3. Error transformation is correct
            
        Test Flow:
            1. Define function with expected error
            2. Configure handler for specific exceptions
            3. Verify error handling behavior
        """
        logger.info(
            "Testing expected exceptions - Expect to see 'Key not found' error"
        )
        @handle_exceptions(
            error_message="Index error",
            expected_exceptions=(KeyError, ValueError),
            raise_exception=IndexError
        )
        def failing_index_function():
            raise KeyError("Key not found")

        with self.assertRaises(IndexError) as context:
            failing_index_function()
        self.assertIn("Index error", str(context.exception))

    def test_handle_exceptions_unexpected_exception(self):
        """
        Test handling of unexpected exception types.
        
        This test verifies that:
            1. Unexpected exceptions pass through unchanged
            2. Expected exception handling still works
            3. Original error context is preserved
            
        Test Flow:
            1. Define function with unexpected error
            2. Configure handler for different error type
            3. Verify original error is raised
        """
        logger.info(
            "Testing unexpected exception - Expect to see a TypeError"
        )
        @handle_exceptions(
            error_message="Query error",
            expected_exceptions=(ValueError,),
            raise_exception=QueryError
        )
        def failing_query_function():
            raise TypeError("Wrong type")

        with self.assertRaises(TypeError) as context:
            failing_query_function()
        self.assertIn("Wrong type", str(context.exception))

    def test_handle_exceptions_successful_execution(self):
        """
        Test behavior when no exceptions occur.
        
        This test verifies that:
            1. Normal execution is not affected
            2. Return values are preserved
            3. No unnecessary error handling occurs
            
        Test Flow:
            1. Define function with successful execution
            2. Verify normal operation
            3. Check return value preservation
        """
        logger.info(
            "Testing successful execution - NO errors should be seen"
        )
        @handle_exceptions(error_message="Should not see this")
        def successful_function():
            return "Success"

        result = successful_function()
        self.assertEqual(result, "Success")

    def test_handle_exceptions_with_args(self):
        """
        Test error handling with function arguments.
        
        This test verifies that:
            1. Functions with arguments work correctly
            2. Arguments affect error conditions
            3. Error handling preserves function context
            
        Test Flow:
            1. Define parameterized function
            2. Test both success and failure cases
            3. Verify error handling with arguments
        """
        logger.info(
            "Testing function with arguments - Expect error when arguments don't match"
        )
        @handle_exceptions(
            error_message="Processing error",
            raise_exception=EmbeddingError
        )
        def function_with_args(arg1, arg2):
            if arg1 != arg2:
                raise ValueError("Arguments don't match")
            return "Match"

        # Test successful case
        result = function_with_args(1, 1)
        self.assertEqual(result, "Match")

        # Test failure case
        with self.assertRaises(EmbeddingError) as context:
            function_with_args(1, 2)
        self.assertIn("Processing error", str(context.exception))

    def test_error_hierarchy(self):
        """
        Test exception class hierarchy relationships.
        
        This test verifies that:
            1. Exception hierarchy is properly structured
            2. Custom exceptions inherit correctly
            3. Type relationships are maintained
            
        Test Flow:
            1. Verify inheritance relationships
            2. Check type hierarchies
            3. Validate exception categorization
        """
        logger.info(
            "Testing error hierarchy - NO errors should be seen"
        )
        self.assertTrue(issubclass(DocumentProcessingError, RAGError))
        self.assertTrue(issubclass(IndexError, RAGError))
        self.assertTrue(issubclass(QueryError, RAGError))
        self.assertTrue(issubclass(EmbeddingError, RAGError))

    def test_error_messages(self):
        """
        Test error message formatting and composition.
        
        This test verifies that:
            1. Error messages are properly formatted
            2. Base and specific messages are combined
            3. Message context is preserved
            
        Test Flow:
            1. Generate error with multiple message components
            2. Verify message composition
            3. Check message formatting
        """
        logger.info(
            "Testing error message format - Expect to see base and specific error messages"
        )
        @handle_exceptions(error_message="Base message")
        def error_message_function():
            raise ValueError("Specific error")

        with self.assertRaises(RAGError) as context:
            error_message_function()
        error_message = str(context.exception)
        self.assertIn("Base message", error_message)
        self.assertIn("Specific error", error_message)

    def tearDown(self):
        """
        Cleanup method that runs after each test.
        
        Performs:
            1. Log test completion
            2. Reset any error handling state
            
        Note:
            Most cleanup is handled automatically as we're
            not creating persistent resources.
        """
        logger.info(
            "=== Test completed ===\n"
        )

if __name__ == '__main__':
    unittest.main() 