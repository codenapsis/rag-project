from src.ingestion.data_ingestion_manager import DataIngestionManager

from src.index.index_manager import IndexManager  # Replace with your IndexManager class path


def main():
    # Step 1: Initialize the IndexManager
    storage_path = "test_index_storage"
    manager = IndexManager(storage_path)
    print("Initialized IndexManager.")

    # Step 2: Create an index
    documents = [
        "Artificial Intelligence is shaping the future.",
        "Python is widely used in data science.",
        "Semantic search is becoming more important.",
        "LlamaIndex helps with retrieval-augmented generation.",
    ]

    # Step 4: Load the index
    index = None
    try:
        index = manager.load_index()
        print("Index loaded successfully.")
    except Exception as e:
        print(f"Failed to load index: {e}")


    # Initialize the DataIngestionManager
    ingestion_manager = DataIngestionManager(manager.get_embbed_model())

    # Configure the pipeline for the given index
    pipeline = ingestion_manager.configure_pipeline(docstore=index.docstore)

    ingestion_manager.ingest_from_texts(texts=documents, pipeline=pipeline)

    # Step 3: Save the index
    try:
        manager.save_index()
        print(f"Index saved to {storage_path}.")
    except Exception as e:
        print(f"Failed to save index: {e}")


if __name__ == "__main__":
    main()