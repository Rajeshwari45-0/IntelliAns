
import re


def split_text(text, chunk_size=500, overlap=100):

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Overlap must be non-negative and smaller than chunk size.")

    # Step 1: Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks = []
    current_chunk = ""

    # Step 2: Add complete sentences to chunks
    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        # Handle sentences longer than chunk_size
        if len(sentence) > chunk_size:

            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            for start in range(0, len(sentence), chunk_size - overlap):
                part = sentence[start:start + chunk_size]
                if part:
                    chunks.append(part)

            continue

        # Check whether sentence fits in current chunk
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:

            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence

        else:

            # Save current chunk
            if current_chunk:
                chunks.append(current_chunk.strip())

            # Start next chunk with previous overlap
            words = current_chunk.split()
            overlap_text = ""

            for word in reversed(words):

                candidate = word + (" " + overlap_text if overlap_text else "")

                if len(candidate) > overlap:
                    break

                overlap_text = candidate

            current_chunk = (
                overlap_text + " " + sentence
                if overlap_text
                else sentence
            )

    # Step 3: Save remaining text
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def main():

    sample_text = """
    IntelliAns is an AI-powered knowledge assistant.
    It allows users to upload documents and ask questions.
    The system extracts text from documents.
    It divides large documents into smaller chunks.
    These chunks will later be converted into embeddings.
    The embeddings help the system find relevant information.
    """

    chunks = split_text(sample_text, chunk_size=100, overlap=20)

    print("Total chunks:", len(chunks))

    for index, chunk in enumerate(chunks, start=1):

        print(f"\n--- Chunk {index} ---")
        print(chunk)

        print("Chunk length:", len(chunk))


if __name__ == "__main__":
    main()
