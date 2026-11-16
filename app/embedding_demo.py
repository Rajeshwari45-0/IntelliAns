
from sentence_transformers import SentenceTransformer, util


def main():

    print("===== IntelliAns Embedding Demo =====")

    # Step 1: Load the embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Step 2: Prepare example sentences
    sentences = [
        "Artificial Intelligence helps computers learn.",
        "AI allows machines to perform intelligent tasks.",
        "I enjoy eating pizza."
    ]

    # Step 3: Convert sentences into embeddings
    embeddings = model.encode(sentences)

    # Step 4: Display embedding information
    print("\nNumber of sentences:", len(sentences))
    print("Embedding shape:", embeddings.shape)

    # Step 5: Compare sentence meanings
    similarity_1 = util.cos_sim(
        embeddings[0],
        embeddings[1]
    )

    similarity_2 = util.cos_sim(
        embeddings[0],
        embeddings[2]
    )

    print("\nSimilarity between sentence 1 and 2:")
    print(similarity_1.item())

    print("\nSimilarity between sentence 1 and 3:")
    print(similarity_2.item())


if __name__ == "__main__":
    main()
