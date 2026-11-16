
from sentence_transformers import SentenceTransformer, util


def semantic_search(query, documents, model):

    # Step 1: Convert document chunks into embeddings
    document_embeddings = model.encode(documents)

    # Step 2: Convert the user's question into an embedding
    query_embedding = model.encode(query)

    # Step 3: Compare question with all document embeddings
    similarities = util.cos_sim(
        query_embedding,
        document_embeddings
    )[0]

    # Step 4: Sort results by similarity score
    results = []

    for index, score in enumerate(similarities):

        results.append({
            "text": documents[index],
            "score": score.item()
        })

    results = sorted(
        results,
        key=lambda item: item["score"],
        reverse=True
    )

    return results


def main():

    print("===== IntelliAns Semantic Search =====")

    # Load the embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Example document chunks
    documents = [
        "IntelliAns is an AI-powered knowledge assistant.",
        "It allows users to upload documents and ask questions.",
        "The system extracts text from PDF documents.",
        "Embeddings convert text into numerical vectors.",
        "Semantic search retrieves information based on meaning.",
        "Python is a popular programming language.",
        "Machine learning helps computers learn from data."
    ]

    # Ask a question
    query = "How can users find information from documents?"

    print("\nUser Question:", query)

    # Search for relevant chunks
    results = semantic_search(query, documents, model)

    print("\n===== Search Results =====")

    for index, result in enumerate(results[:3], start=1):

        print(f"\nResult {index}")
        print("Similarity:", round(result["score"], 4))
        print("Text:", result["text"])


if __name__ == "__main__":
    main()
