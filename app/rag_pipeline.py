
import os
import json
import time

import faiss
import numpy as np

from dotenv import load_dotenv
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer


def main():

    print("===== IntelliAns AI Knowledge Assistant =====")

    # STEP 1: Load Gemini API key from .env
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("\nGemini API key not found.")
        print("Please check your .env file.")
        return

    # Create Gemini client with a 30-second timeout
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=120000
        )
    )

    # STEP 2: Load saved FAISS index and document metadata
    index_path = "vector_store/pdf_index.index"
    documents_path = "vector_store/pdf_documents.json"

    if not os.path.exists(index_path):
        print("\nFAISS index not found.")
        print("Run: python -m app.build_pdf_index")
        return

    if not os.path.exists(documents_path):
        print("\nDocument metadata file not found.")
        return

    index = faiss.read_index(index_path)

    with open(documents_path, "r", encoding="utf-8") as file:
        documents = json.load(file)

    print("\nKnowledge base loaded!")
    print("Total vectors:", index.ntotal)

    if index.ntotal == 0:
        print("\nThe PDF index is empty.")
        return

    # STEP 3: Load sentence embedding model
    print("\nLoading embedding model...")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # STEP 4: Get question from user
    query = input("\nAsk a question about your PDF: ").strip()

    if not query:
        print("\nPlease enter a question.")
        return

    # STEP 5: Convert question into an embedding
    print("\nConverting question into embedding...")

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    )

    # STEP 6: Retrieve relevant document chunks
    print("\nSearching for relevant information...")

    top_k = min(3, index.ntotal)

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    retrieved_chunks = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1 or idx >= len(documents):
            continue

        document = documents[idx]

        retrieved_chunks.append({
            "text": document["text"],
            "filename": document["filename"],
            "page_number": document["page_number"],
            "chunk_number": document["chunk_number"],
            "similarity": float(score)
        })

    if not retrieved_chunks:
        print("\nNo relevant information found.")
        return

    # STEP 7: Prepare context from retrieved chunks
    context_parts = []

    for number, chunk in enumerate(retrieved_chunks, start=1):

        context_parts.append(
            f"""
Source {number}
Filename: {chunk['filename']}
Page: {chunk['page_number']}
Content:
{chunk['text']}
"""
        )

    context = "\n".join(context_parts)

    # STEP 8: Create the RAG prompt
    prompt = f"""
You are IntelliAns, an intelligent document question-answering assistant.

Answer the user's question using only the information
provided in the document context.

Instructions:
1. Give a clear and well-structured answer.
2. Explain technical terms in simple language.
3. Combine information from multiple sources when useful.
4. Do not invent facts or use outside knowledge.
5. If the context does not contain enough information,
   clearly say that the document does not provide enough information.
6. Mention relevant PDF page numbers in your answer.
7. Avoid repeating irrelevant information.
8. If an abbreviation is explained in the context,
   provide its full form and explanation.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{query}

ANSWER:
"""

    # STEP 9: Generate answer using Gemini with retry handling
    print("\nGenerating answer using Gemini...")

    answer = None

    for attempt in range(3):

        try:

            print(f"\nGemini attempt {attempt + 1}/3...")

            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt
            )

            answer = response.text

            if answer:
                break

            print("Gemini returned an empty response.")

        except Exception as error:

            error_message = str(error)

            print("\nGemini request failed.")
            print("Error:", error_message)

            temporary_error = any(
                term in error_message.upper()
                for term in [
                    "503",
                    "UNAVAILABLE",
                    "429",
                    "RESOURCE_EXHAUSTED",
                    "TIMEOUT",
                    "TIMED OUT",
                    "DEADLINE_EXCEEDED"
                ]
            )

            if temporary_error and attempt < 2:

                wait_time = 5 * (2 ** attempt)

                print(
                    f"Gemini may be temporarily unavailable. "
                    f"Retrying in {wait_time} seconds..."
                )

                time.sleep(wait_time)

            else:
                break

    if not answer:

        print("\nUnable to generate an answer.")
        print("Please check your Gemini API access and try again.")
        return

    # STEP 10: Display generated answer
    print("\n========== IntelliAns Answer ==========\n")

    print(answer)

    # STEP 11: Display retrieved source references
    print("\n========== Retrieved Sources ==========")

    for number, chunk in enumerate(retrieved_chunks, start=1):

        print(f"\nSource {number}")
        print("Filename:", chunk["filename"])
        print("Page:", chunk["page_number"])
        print("Chunk:", chunk["chunk_number"])
        print("Similarity:", round(chunk["similarity"], 4))

    print("\nRAG pipeline completed successfully!")


if __name__ == "__main__":
    main()
