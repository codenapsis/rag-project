from src.index.index_manager import IndexManager

def main():
    # Step 1: Initialize IndexManager
    storage_path = "empty_index_storage"
    print("Initializing IndexManager...")
    manager = IndexManager(storage_path)
    print("IndexManager initialized successfully.")

    # Step 2: Create an empty index
    print("Creating an empty index...")
    manager.create_index([])  # Pass an empty list to create an empty index
    print("Empty index created successfully.")

    # Step 3: Save the empty index
    print(f"Saving the empty index to {storage_path}...")
    manager.save_index()
    print("Empty index saved successfully.")

    # Step 4: Load the empty index
    print(f"Loading the empty index from {storage_path}...")
    manager.load_index()
    print("Empty index loaded successfully.")

if __name__ == "__main__":
    main()
