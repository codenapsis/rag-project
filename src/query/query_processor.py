from typing import List, Optional
from src.pipeline.rag_pipeline import RAGPipeline
from src.index.index_manager import IndexManager
import logging
from src.utils.error_handler import handle_exceptions, QueryError

logger = logging.getLogger(__name__)

class QueryProcessor:
    """
    QueryProcessor is like a smart assistant that helps you ask questions about your documents
    and get relevant answers.

    What does it do?
    1. Takes your questions
    2. Searches through your documents
    3. Returns the most relevant information
    4. Provides alternative search methods if needed

    Think of it as:
    - A librarian who knows exactly where to find information
    - A search engine for your specific documents
    - A helper that understands what you're looking for

    Example Usage:
        # Create a processor with your index
        processor = QueryProcessor(my_index_manager)
        
        # Ask a question
        answer = processor.process_query("What is machine learning?")
        print(answer)
        
        # If you need more results, try the alternative search
        more_results = processor.process_query_bm25("machine learning examples")
    """

    def __init__(self, index_manager: IndexManager):
        """
        Sets up the QueryProcessor with your document index.

        Args:
            index_manager (IndexManager): The manager containing your indexed documents
                This is like giving the processor access to your library of documents

        Example:
            index_manager = IndexManager("my_documents")
            processor = QueryProcessor(index_manager)
        """
        self.index_manager = index_manager
        self.pipeline = RAGPipeline()
        
        # Initialize the search pipeline with your index
        if self.index_manager.index:
            self.pipeline.initialize_pipeline(self.index_manager.index)
            logger.info("QueryProcessor initialized and ready to answer questions")
        else:
            logger.warning("Index not found. Please ensure index is loaded.")

    @handle_exceptions(
        error_message="Failed to process query",
        expected_exceptions=(ValueError, Exception),
        raise_exception=QueryError
    )
    def process_query(self, query: str) -> List[str]:
        """
        Searches for information to answer your question.

        How it works:
        1. Takes your question
        2. Searches through the indexed documents
        3. Finds the most relevant pieces of information
        4. Returns these pieces as a list

        Args:
            query (str): Your question or what you want to know about
                Example: "What is Python used for?"
                Example: "How does machine learning work?"

        Returns:
            List[str]: List of relevant text passages that answer your question
                Note: Returns empty list if no relevant information is found

        Raises:
            ValueError: If the system isn't ready to answer questions

        Example:
            processor = QueryProcessor(my_index_manager)
            
            # Ask a question
            results = processor.process_query("What is deep learning?")
            
            # Print the answers
            for i, result in enumerate(results, 1):
                print(f"Answer {i}: {result}")
        """
        if not self.pipeline:
            raise ValueError(
                "Query processor not ready. "
                "Please ensure the index is properly loaded."
            )

        logger.info(f"Processing query: '{query}'")
        results = self.pipeline.run_pipeline(query)
        
        if results:
            logger.info(f"Found {len(results)} relevant answers")
            for i, result in enumerate(results, 1):
                logger.debug(f"Result {i}: {result[:100]}...")
        else:
            logger.info("No relevant information found")
        
        return results

    @handle_exceptions(
        error_message="Failed to process BM25 query",
        raise_exception=QueryError
    )
    def process_query_bm25(self, query: str) -> List[str]:
        """
        Alternative search method using keywords (BM25 algorithm).

        When to use this?
        - When the regular search doesn't find what you need
        - When you want to search by specific words
        - When you want a different perspective on the search

        Args:
            query (str): Your question or keywords
                Example: "Python programming tutorial"
                Example: "machine learning examples"

        Returns:
            List[str]: List of relevant text passages
                Note: Results might be different from regular search

        Example:
            processor = QueryProcessor(my_index_manager)
            
            # Search using keywords
            results = processor.process_query_bm25("Python examples")
            
            # Print what was found
            for i, result in enumerate(results, 1):
                print(f"Found text {i}: {result}")
        """
        if not self.pipeline:
            raise ValueError(
                "Query processor not ready. "
                "Please ensure the index is properly loaded."
            )

        logger.info(f"Processing BM25 query: '{query}'")
        results = self.pipeline.run_pipeline_bm25(query)
        
        if results:
            logger.info(f"BM25 search found {len(results)} matches")
        else:
            logger.info("No matches found with BM25 search")
        
        return results

    def query_index(self, index, query: str):
        pass

    def retrieve_context(self, query: str, index):
        pass

    def format_response(self, llm_response: str, context: str):
        pass