from functools import wraps
import logging
from typing import Type, Union, Callable
import traceback

logger = logging.getLogger(__name__)

class RAGError(Exception):
    """Base exception class for RAG system errors"""
    pass

class DocumentProcessingError(RAGError):
    """Raised when there's an error processing documents"""
    pass

class IndexError(RAGError):
    """Raised when there's an error with the index operations"""
    pass

class QueryError(RAGError):
    """Raised when there's an error processing queries"""
    pass

class EmbeddingError(RAGError):
    """Raised when there's an error with embeddings"""
    pass

def handle_exceptions(
    error_message: str = "An error occurred",
    expected_exceptions: tuple = (Exception,),
    raise_exception: Type[Exception] = RAGError
) -> Callable:
    """
    A decorator for standardized error handling across the RAG system.

    Args:
        error_message (str): Base error message to use
        expected_exceptions (tuple): Exceptions to catch
        raise_exception (Type[Exception]): Exception type to raise

    Example:
        @handle_exceptions(error_message="Failed to process document")
        def process_document(self, doc):
            # Process document here
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except expected_exceptions as e:
                logger.error(
                    f"{error_message}: {str(e)}\n"
                    f"Traceback: {traceback.format_exc()}"
                )
                raise raise_exception(f"{error_message}: {str(e)}") from e
        return wrapper
    return decorator 