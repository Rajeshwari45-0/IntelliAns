
import os
import json
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer


def main():

    print("===== IntelliAns PDF Search =====")

    # Step 1: Define paths to saved files
    index_path = "vector_store/pdf_index.index"
    documents_path = "vector_store/pdf_documents.json"

    # Step 2: Check whether saved files exist
    if not os.path.exists(index_path):
        print("FAISS index not found. Run build_pdf_index.py first.")
        return

    if not os.path.exists(documents_path):
        print("PDF document data not found.")
        return

    # Step 3: Load the saved FAISS index
    index = faiss.read_index(index_path)

    # Step 4: Load PDF chunks and their metadata
    with open(documents_path, "r", encoding="utf-8") as file:
        documents = json.load(file)

    print("\nPDF index loaded successfully!")
    print("Total vectors:", index.ntotal)
    print("Total document chunks:", len(documents))

    # Step 5: Load the same embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Step 6: Get a question from the user
    query = input("\nEnter your question about the PDF: ").strip()

    if not query:
        print("Please enter a question.")
        return

    # Step 7: Convert the question into an embedding
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    )

    # Step 8: Search FAISS for the 3 most relevant chunks
    top_k = min(3, index.ntotal)

    scores, indices = index.search(query_embedding, top_k)

    # Step 9: Display matching chunks and their source metadata
    print("\n===== Relevant PDF Results =====")

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):

        if idx == -1:
            continue

        document = documents[idx]

        print(f"\nResult {rank}")
        print("Similarity:", round(float(score), 4))
        print("Filename:", document["filename"])
        print("Page number:", document["page_number"])
        print("Chunk number:", document["chunk_number"])
        print("Text:", document["text"])

    print("\nPDF search completed successfully!")


if __name__ == "__main__":
    main()
