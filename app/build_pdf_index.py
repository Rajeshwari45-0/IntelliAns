
import os
import json
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from app.process_document import process_document


def main():

    print("===== IntelliAns PDF Vector Index Builder =====")

    # Step 1: Select the PDF
    pdf_path = "data/sample.pdf"

    if not os.path.exists(pdf_path):
        print("PDF file not found:", pdf_path)
        return

    # Step 2: Process the PDF
    print("\nProcessing PDF...")

    chunks = process_document(pdf_path)

    if not chunks:
        print("No text chunks found in the PDF.")
        return

    print("Total chunks created:", len(chunks))

    # Step 3: Extract text from chunks
    documents = [chunk["text"] for chunk in chunks]

    # Step 4: Load embedding model
    print("\nLoading embedding model...")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Step 5: Convert chunks into embeddings
    print("\nGenerating embeddings...")

    embeddings = model.encode(
        documents,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    # Step 6: Create FAISS index
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print("\nTotal vectors stored:", index.ntotal)

    # Step 7: Create storage folder
    os.makedirs("vector_store", exist_ok=True)

    # Step 8: Save FAISS index
    faiss.write_index(
        index,
        "vector_store/pdf_index.index"
    )

    # Step 9: Save chunk text and metadata
    with open(
        "vector_store/pdf_documents.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(chunks, file, indent=4, ensure_ascii=False)

    print("\nPDF index created successfully!")
    print("FAISS index: vector_store/pdf_index.index")
    print("Document metadata: vector_store/pdf_documents.json")


if __name__ == "__main__":
    main()
