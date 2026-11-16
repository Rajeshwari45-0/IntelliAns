
import os
import json
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer


def main():

    print("===== IntelliAns Saved Vector Search =====")

    # Step 1: Define saved file paths
    index_path = "vector_store/intellians.index"
    documents_path = "vector_store/documents.json"

    # Step 2: Check whether saved files exist
    if not os.path.exists(index_path):
        print("FAISS index not found. Run faiss_search.py first.")
        return

    if not os.path.exists(documents_path):
        print("Document metadata file not found.")
        return

    # Step 3: Load the previously saved FAISS index
    index = faiss.read_index(index_path)

    print("\nSaved FAISS index loaded successfully!")

    # Step 4: Load original document chunks
    with open(documents_path, "r", encoding="utf-8") as file:
        documents = json.load(file)

    print("Total vectors:", index.ntotal)
    print("Total documents:", len(documents))

    # Step 5: Load the same embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Step 6: Ask the user for a question
    query = input("\nEnter your question: ")

    # Step 7: Convert question into an embedding
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    )

    # Step 8: Search the saved FAISS index
    top_k = min(3, index.ntotal)

    scores, indices = index.search(query_embedding, top_k)

    # Step 9: Display matching document chunks
    print("\n===== Search Results =====")

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):

        if idx == -1:
            continue

        print(f"\nResult {rank}")
        print("Similarity:", round(float(score), 4))
        print("Text:", documents[idx])

    print("\nSearch completed successfully!")


if __name__ == "__main__":
    main()
