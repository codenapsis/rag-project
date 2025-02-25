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
    """Test suite for error handler functionality"""

    def setUp(self):
        """
        Executed before each test.
        Informs that the following errors are expected.
        """
        logger.info(
            "\n=== NOTE: The following errors are expected and part of the tests ==="
        )

    def test_handle_exceptions_basic(self):
        """Test basic error handling functionality"""
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
        """Test with custom exception type"""
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
        """Test with specific expected exceptions"""
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
        """Test with unexpected exception type"""
        logger.info(
            "Testing unexpected exception - Expect to see a TypeError"
        )
        @handle_exceptions(
            error_message="Query error",
            expected_exceptions=(ValueError,),
            raise_exception=QueryError
        )
        def failing_query_function():
            raise TypeError("Wrong type")  # Unexpected exception type

        with self.assertRaises(TypeError) as context:
            failing_query_function()
        self.assertIn("Wrong type", str(context.exception))

    def test_handle_exceptions_successful_execution(self):
        """Test when no exception is raised"""
        logger.info(
            "Testing successful execution - NO errors should be seen"
        )
        @handle_exceptions(error_message="Should not see this")
        def successful_function():
            return "Success"

        result = successful_function()
        self.assertEqual(result, "Success")

    def test_handle_exceptions_with_args(self):
        """Test with function arguments"""
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
        """Test error class hierarchy"""
        logger.info(
            "Testing error hierarchy - NO errors should be seen"
        )
        self.assertTrue(issubclass(DocumentProcessingError, RAGError))
        self.assertTrue(issubclass(IndexError, RAGError))
        self.assertTrue(issubclass(QueryError, RAGError))
        self.assertTrue(issubclass(EmbeddingError, RAGError))

    def test_error_messages(self):
        """Test error messages are properly formatted"""
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
        Executed after each test.
        Indicates test completion.
        """
        logger.info(
            "=== Test completed ===\n"
        )

if __name__ == '__main__':
    unittest.main() 