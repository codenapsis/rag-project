from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.storage.docstore import SimpleDocumentStore

from src.ingestion.data_ingestion_manager import DataIngestionManager
from src.pipeline.rag_pipeline import RAGPipeline
from src.index.index_manager import IndexManager
from src.embeddings.embedding_manager import EmbeddingManager
from src.documents.document_processor import DocumentProcessor

def main():
    # Initialize the IndexManager to manage the index
    storage_path = "C:\workspace\\rag-project\src\data"
    index_manager = IndexManager(storage_path=storage_path)
    embedding_model = EmbeddingManager().load_embedding_model()
    Settings.embed_model = embedding_model
    Settings.llm = None

    # Step 1: Create a new index
    print("Creating a new index...")
    index_manager.create_index([], embedding_model)
    index = index_manager.load_index()

    # Step 3: Initialize the DataIngestionManager
    ingestion_manager = DataIngestionManager()

    # Step 4: Configure the pipeline for ingestion
    pipeline = ingestion_manager.configure_pipeline(docstore=index.docstore, embedding_model=embedding_model)

    # Step 5: Ingest data into the index
    print("Ingesting data into the index...")
    raw_texts = [
        "Artificial Intelligence is transforming industries.",
        "Machine learning enables predictive analytics.",
        "Deep learning allows machines to learn from data.",
        "Artificial intelligence (AI) is the simulation of human intelligence in machines designed to perform tasks that typically require human cognition.",
        "AI systems can analyze vast amounts of data to identify patterns and make decisions faster than humans.",
        "Machine learning, a subset of AI, enables systems to learn from data and improve their performance over time.",
        "AI-powered tools are transforming industries like healthcare, finance, and transportation by automating complex processes.",
        "Natural language processing (NLP) allows AI to understand and generate human language.",
        "Computer vision, an AI field, enables machines to interpret and analyze visual data such as images and videos.",
        "AI is being used in autonomous vehicles to process sensor data and make real-time driving decisions.",
        "Personalized recommendations on platforms like Netflix and Amazon are powered by AI algorithms.",
        "AI chatbots and virtual assistants like ChatGPT and Alexa are revolutionizing customer service and communication.",
        "Deep learning, a type of machine learning, uses neural networks to solve complex problems.",
        "AI can help detect fraudulent activities in financial transactions by analyzing anomalies in data patterns.",
        "Robotics combined with AI is enabling the creation of intelligent machines that can perform physical tasks.",
        "AI is being applied in medical imaging to assist doctors in diagnosing diseases such as cancer.",
        "Ethical concerns about AI include issues related to bias, transparency, and the potential loss of jobs.",
        "AI research aims to create general artificial intelligence, capable of performing any intellectual task a human can."
    ]

    document_processor = DocumentProcessor()
    documents = document_processor.create_documents(raw_texts)
    documents = document_processor.batch_add_embeddings(embedding_model, documents)

    ingestion_manager.ingest_from_documents(documents, pipeline=pipeline)
    print("Data ingestion completed.")

    print(f"Number of documents in the index: {len(index.docstore.docs)}")
    for doc_id, doc in index.docstore.docs.items():
        print(f"Doc ID: {doc_id}, Text: {doc.text[:100]}, Emb: {doc.embedding}")

    index_manager.save_index()

    docstore = SimpleDocumentStore.from_persist_path(persist_path=storage_path+"\docstore.json")

    # Obtener todos los IDs de los documentos almacenados
    all_doc_ids = list(docstore.docs.keys())

    # Recuperar todos los nodos utilizando sus IDs
    nodes = docstore.get_nodes(all_doc_ids)

    index = index_manager.get_index(nodes, embedding_model)

    # Realizar una consulta
    query = "Para que sirve la inteligencia artificial?"

    # Step 6: Test the RAGPipeline
    print("Initializing RAGPipeline...")
    rag_pipeline = RAGPipeline()
    rag_pipeline.initialize_pipeline(index)

    # Step 7: Run a query and retrieve documents
    
    print(f"Running query: '{query}'")
    results = rag_pipeline.run_pipeline(query)
    print("Retrieved Documents:")
    for result in results:
        print("-", result)


if __name__ == "__main__":
    main()
