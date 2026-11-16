
import os
import json
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer


def main():

    print("===== IntelliAns FAISS Vector Search =====")

    # Step 1: Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Step 2: Prepare sample document chunks
    documents = [
        "IntelliAns is an AI-powered knowledge assistant.",
        "It allows users to upload documents and ask questions.",
        "The system extracts text from PDF documents.",
        "Embeddings convert text into numerical vectors.",
        "Semantic search retrieves information based on meaning.",
        "Python is a popular programming language.",
        "Machine learning helps computers learn from data."
    ]

    # Step 3: Convert documents into normalized embeddings
    document_embeddings = model.encode(
        documents,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # FAISS requires float32 vectors
    document_embeddings = np.asarray(
        document_embeddings,
        dtype=np.float32
    )

    # Step 4: Create a FAISS index
    embedding_dimension = document_embeddings.shape[1]

    index = faiss.IndexFlatIP(embedding_dimension)

    # Step 5: Add document vectors to the index
    index.add(document_embeddings)

    print("\nTotal vectors stored:", index.ntotal)

    # Step 6: Convert user question into an embedding
    query = "How can users find information from documents?"

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    )

    # Step 7: Search for the top 3 matching vectors
    scores, indices = index.search(query_embedding, 3)

    print("\nUser Question:", query)

    print("\n===== FAISS Search Results =====")

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):

        print(f"\nResult {rank}")
        print("Similarity:", round(float(score), 4))
        print("Text:", documents[idx])

    # Step 8: Save the FAISS index and document metadata
    os.makedirs("vector_store", exist_ok=True)

    faiss.write_index(
        index,
        "vector_store/intellians.index"
    )

    with open(
        "vector_store/documents.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(documents, file, indent=4)

    print("\nFAISS index and documents saved successfully!")


if __name__ == "__main__":
    main()
