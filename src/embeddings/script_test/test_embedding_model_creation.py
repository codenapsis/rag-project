from src.ingestion.data_ingestion_manager import DataIngestionManager
from src.pipeline.rag_pipeline import RAGPipeline
from src.index.index_manager import IndexManager
from src.embeddings.embedding_manager import EmbeddingManager


def main():

    embedding_manager = EmbeddingManager()
    embedding_model = embedding_manager.load_embedding_model()
    print(embedding_model)
    string = "Hi"
    embedding = embedding_manager.get_embeddings(string)
    print(embedding)




if __name__ == "__main__":
    main()
