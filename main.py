"""
RAG System Main Application.

This module serves as the entry point for the RAG (Retrieval-Augmented Generation) system.
It implements the core RAG workflow demonstrated in the integration tests, providing
a production-ready implementation for document processing and query answering.

Key Features:
    - Complete RAG pipeline implementation
    - Document processing and embedding
    - Index creation and persistence
    - Query processing and response generation
    
System Components:
    - RAGPipeline: Core query processing
    - IndexManager: Document storage and retrieval
    - EmbeddingManager: Vector embeddings
    - DocumentProcessor: Text processing
    - LLMService: Response generation using Azure-hosted LLM
    
Usage:
    python main.py

TODO for Students:
    You need to implement the LLM integration in this system. The following steps are required:
    1. Import and initialize the LLMService from src.llm.llm_service
    2. Use the LLMService methods to:
       - create_context: Convert the retrieved documents into a context string
       - prompt_engineering: Create a well-structured prompt
       - generate_response: Get the final response from the LLM
"""

import logging
import os
import shutil
# TODO for Students: You will need to import the necessary components
# RAGPipeline
# IndexManager
# EmbeddingManager
# DocumentProcessor
# LLMService
from llama_index.core import Document, Settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGSystem:
    """
    Main RAG system implementation.
    
    This class implements the production version of the RAG system,
    following the architecture validated in integration tests.
    """
    
    def __init__(self, storage_path="data"):
        """
        Initialize RAG system components.
        
        Args:
            storage_path (str): Path for document storage
        """
        logger.info("Initializing RAG system...")
        
        # Configure Settings
        Settings.llm = None  # Disable LLM for testing
        Settings.context_window = 2048
        Settings.num_output = 256
        
        # Initialize components
        self.storage_path = storage_path
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)
            
        # TODO for Students: Initialize the system components
        # 1. Initialize the RAG Pipeline
        # self.rag_pipeline = 
        
        # 2. Initialize the document processing components
        # self.index_manager = 
        # self.embedding_manager = 
        # self.document_processor = 
        # self.embed_model = 
        
        logger.info("All components initialized")
        
    def process_documents(self, raw_texts):
        """
        Process raw texts into searchable documents.
        
        Args:
            raw_texts (List[str]): List of text content to process
            
        Returns:
            Index: Searchable document index
        
        TODO for Students:
        1. Create Document objects from raw texts
           - Use the document processor to convert the raw texts into Document objects
           - The document processor has a method for creating documents from text
        
        2. Add embeddings to the documents
           - Documents need vector representations (embeddings) to be searchable
           - The document processor can add embeddings in batch mode using your embedding model
        
        3. Create and save the index
           - Use the index manager to create a searchable index from your documents
           - Don't forget to save the index for persistence
        
        4. Log the process
           - Log important information like the number of documents processed
           - Include when the index is created and saved
        
        5. Return the index
           - The index will be needed later for querying
        
        Optional Enhancements:
        - Error handling: What happens if document processing fails?
        - Chunking: How would you handle very large documents?
        - Metadata: Consider adding metadata to improve document filtering
        
        Note: This function should process the raw texts into an indexed, searchable format.
        The document processor and index manager handle most of the complexity.
        """
        pass
        
    def query_documents(self, index, query):
        """
        Process a query against the document index.
        
        Args:
            index: Document index to query
            query (str): Query text
            
        Returns:
            List[str]: Query results
        
        TODO for Students:
        1. Initialize the RAG Pipeline if not already done
           - Check if you already have a pipeline instance
           - If not, create one and initialize it with your index
        
        2. Run the query through the pipeline
           - The pipeline has a method to process queries and return relevant documents
        
        3. Log the results
           - Log useful information about the search results
        
        4. Return the results
           - Return the list of relevant document texts
        
        Optional Enhancements:
        - Error handling: What if the pipeline fails?
        - Logging: Consider logging previews of the results
        - Edge cases: How would you handle empty queries or no results?
        
        Note: This function should connect the pipeline to the index and run the query.
        The RAG pipeline handles the search logic.
        """
        pass

def process_with_llm(results: list, query: str):
    """
    Process the RAG results using the LLM to generate a final response.
    
    This function coordinates the LLM processing pipeline:
    1. Takes the results from the RAG system (relevant documents)
    2. Creates a unified context from these documents
    3. Generates a well-structured prompt
    4. Gets the LLM to generate a response
    
    Args:
        results (List[str]): Results from the RAG system. These are the most relevant
                            documents found for the query.
        query (str): Original user query that needs to be answered.
        
    Returns:
        str: Generated response from the LLM, which should be a coherent answer
             based on the provided context and query.
    
    TODO for Students:
    1. Initialize the LLMService class
       - This will be your interface to the Azure-hosted LLM
    
    2. Use the LLMService methods in the following order:
       a. create_context(results):
          - Combine the RAG results into a single, coherent context string
          - Consider how to handle multiple documents effectively
       
       b. prompt_engineering(context, query):
          - Create a prompt that will guide the LLM to generate a good response
          - Include clear instructions for the LLM
          - Structure the context and query appropriately
       
       c. generate_response(context, query):
          - Use the LLM to generate the final response
          - Handle any potential errors or edge cases
    
    3. Return the generated response
    
    Note: All the implementation logic should be done in the LLMService class methods.
    This function should only coordinate the calls to those methods.
    """
    pass

def main():
    """
    Main application entry point.
    
    Implements the complete RAG workflow:
        1. System initialization
        2. Document processing
        3. Query handling
        4. LLM response generation

    The workflow processes multiple documents to build a knowledge base,
    then runs multiple test queries against this knowledge base.
    For each query, it:
        1. Retrieves relevant documents using RAG
        2. Processes these results with the LLM
        3. Logs both the RAG results and the LLM's response
    """
    try:
        # Initialize system
        rag_system = RAGSystem()
        
        # Example documents
        # TODO for Students: Add more texts to create a larger index
        # Hint: You can use ChatGPT to generate synthetic data. Try these prompts:
        # - "Generate 5 paragraphs about different Python programming concepts"
        # - "Write 3 detailed paragraphs about software testing methodologies"
        # - "Explain object-oriented programming principles in 4 paragraphs"
        # 
        # The more varied and numerous the texts, the better the RAG system will perform.
        # Aim for at least 10-15 different texts covering various aspects of your topics.
        raw_texts = [
            "Python is a high-level programming language known for its simplicity",
            "Python is widely used in data science and machine learning",
            "Software testing is essential for quality assurance",
            # Add more texts here...
        ]
        
        # Process documents and create the searchable index
        # This step vectorizes all documents and creates an efficient search structure
        index = rag_system.process_documents(raw_texts)
        
        # Example queries to test the system
        # TODO for Students: Add more diverse queries to test your system
        # Try queries that:
        # - Ask about specific information: "What are Python's main features?"
        # - Request comparisons: "Compare unit testing and integration testing"
        # - Ask for explanations: "Explain how Python handles memory management"
        # - Seek examples: "Give examples of Python use cases in data science"
        test_queries = [
            "What is Python used for?",
            "Why is testing important?"
            # Add more queries here...
        ]
        
        # Process each query in sequence
        # The system will:
        # 1. Use RAG to find relevant documents for each query
        # 2. Show the retrieved documents (RAG results)
        # 3. Process these results with the LLM to generate a final answer
        # 4. Display both RAG results and the LLM's response for comparison
        for query in test_queries:
            # Get relevant documents using RAG
            results = rag_system.query_documents(index, query)
            logger.info(f"\nQuery: {query}")
            logger.info("RAG Results:")
            for i, result in enumerate(results):
                logger.info(f"Result {i + 1}: {result[:100]}...")
            
            # TODO for Students: Process results with LLM
            # Implement the LLM processing here:
            # 1. Call process_with_llm with the RAG results and query
            # 2. Log the LLM's response to see the final generated answer
            # Example:
            # llm_response = process_with_llm(results, query)
            # logger.info(f"\nLLM Response: {llm_response}")
                
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()