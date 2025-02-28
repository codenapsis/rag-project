from typing import Dict, Any, List
import logging
import sys
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.base.base_query_engine import BaseQueryEngine
from src.utils.error_handler import handle_exceptions, RAGError

logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    RAGPipeline (Retrieval-Augmented Generation Pipeline) is like a smart librarian
    that helps you find and understand information from your documents.

    What is RAG?
    - Retrieval: Finding the most relevant information from your documents
    - Augmented: Adding this information to provide context
    - Generation: Creating meaningful responses based on the found information

    Think of it like this:
    1. You ask a question
    2. The pipeline searches through your documents
    3. It finds the most relevant pieces of information
    4. It returns these relevant pieces as an answer

    Example Usage:
        # Initialize the pipeline
        pipeline = RAGPipeline()
        
        # Set up with your document index
        pipeline.initialize_pipeline(my_index)
        
        # Ask questions
        answer = pipeline.run_pipeline("What is machine learning?")
    """

    def __init__(self):
        """
        Creates a new RAG pipeline.
        
        The pipeline starts empty and needs to be initialized with an index
        before it can answer questions.
        """
        self.index = None
        self.retriever: BaseQueryEngine = None
        self.similarity_cutoff = 0.5  # Minimum relevance score (0 to 1)
        self.similarity_top_k = 2     # Number of results to return
        logger.info("Created new RAG pipeline")

    @handle_exceptions(
        error_message="Failed to initialize pipeline",
        raise_exception=RAGError
    )
    def initialize_pipeline(self, index) -> None:
        """
        Sets up the pipeline with your document index.

        Think of this as:
        - Giving the librarian (pipeline) access to your library (index)
        - Setting up the search rules (how to find relevant information)

        Args:
            index: Your document index (created by IndexManager)
                This contains all the documents the pipeline can search through

        Example:
            pipeline = RAGPipeline()
            pipeline.initialize_pipeline(my_document_index)
            print("Pipeline is ready to answer questions!")
        """
        if index is None:
            raise ValueError("Index cannot be None")
        
        self.index = index
        self.retriever = index.as_query_engine(
            response_mode="no_text",  # Return only relevant document parts
            similarity_top_k=self.similarity_top_k,  # How many results to return
            similarity_cutoff=self.similarity_cutoff  # Minimum relevance score
        )
        logger.info(
            f"Pipeline initialized with settings: "
            f"top_k={self.similarity_top_k}, "
            f"cutoff={self.similarity_cutoff}"
        )

    @handle_exceptions(
        error_message="Failed to run pipeline",
        raise_exception=RAGError
    )
    def run_pipeline(self, query: str) -> List[str]:
        """
        Searches for information related to your question.

        How it works:
        1. Takes your question
        2. Searches through the documents
        3. Finds the most relevant pieces of information
        4. Returns these pieces as a list

        Args:
            query (str): Your question or search term
                Example: "What is Python used for?"

        Returns:
            List[str]: List of relevant text passages found in the documents
                Note: Returns empty list if nothing relevant is found

        Raises:
            ValueError: If the pipeline hasn't been initialized

        Example:
            pipeline = RAGPipeline()
            pipeline.initialize_pipeline(my_index)
            
            # Ask a question
            results = pipeline.run_pipeline("What is deep learning?")
            
            # Print the results
            for i, result in enumerate(results, 1):
                print(f"Result {i}: {result}")
        """
        if not self.retriever:
            raise ValueError("Pipeline not initialized")

        logger.info(f"Processing query: '{query}'")
        
        # Search for relevant information
        results = self.retriever.query(query)
        
        # Process and filter the results
        retrieved_texts = []
        for node in results.source_nodes:
            similarity = getattr(node, 'score', None)
            if similarity:
                logger.info(
                    f"Found relevant text (score: {similarity:.3f})"
                )
                if similarity > self.similarity_cutoff:
                    retrieved_texts.append(node.node.text)
                    logger.debug(
                        f"Added text: {node.node.text[:100]}..."
                    )
                else:
                    logger.debug(
                        f"Skipped text with low relevance: {similarity:.3f}"
                    )
        
        logger.info(f"Found {len(retrieved_texts)} relevant results")
        return retrieved_texts

    @handle_exceptions(
        error_message="Failed to run BM25 pipeline",
        raise_exception=RAGError
    )
    def run_pipeline_bm25(self, query: str) -> List[str]:
        """
        Alternative search method using BM25 (a keyword-based search algorithm).

        When to use this?
        - When the regular search isn't finding what you need
        - When you want to search by specific keywords
        - As a backup search method

        Args:
            query (str): Your question or search term
                Example: "Python programming examples"

        Returns:
            List[str]: List of relevant text passages

        Example:
            pipeline = RAGPipeline()
            pipeline.initialize_pipeline(my_index)
            
            # Try keyword-based search
            results = pipeline.run_pipeline_bm25("Python tutorial")
            
            for result in results:
                print(result)
        """
        if not self.index:
            raise ValueError("Pipeline not initialized")

        logger.info(f"Running BM25 search for: '{query}'")
        
        # Create a keyword-based searcher
        retriever = BM25Retriever.from_defaults(
            docstore=self.index.docstore,
            similarity_top_k=self.similarity_top_k
        )
        
        # Get results
        results = retriever.retrieve(query)
        texts = [result.node.text for result in results]
        
        logger.info(f"BM25 search found {len(texts)} results")
        return texts
