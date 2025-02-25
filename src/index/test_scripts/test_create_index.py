from ..index_manager import IndexManager  # Replace with your IndexManager class path


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
    manager.create_index(documents)
    print("Index created successfully.")

    # Step 3: Save the index
    try:
        manager.save_index()
        print(f"Index saved to {storage_path}.")
    except Exception as e:
        print(f"Failed to save index: {e}")

    # Step 4: Load the index
    try:
        manager.load_index()
        print("Index loaded successfully.")
    except Exception as e:
        print(f"Failed to load index: {e}")


if __name__ == "__main__":
    main()