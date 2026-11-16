
from app.pdf_reader import extract_text_from_pdf
from app.text_cleaner import clean_text
from app.text_chunker import split_text


def process_document(file_path):

    # Step 1: Extract text from PDF
    pages = extract_text_from_pdf(file_path)

    if pages is None:
        return []

    all_chunks = []

    # Step 2: Process each page
    for page in pages:

        page_text = page["text"]

        # Step 3: Clean extracted text
        cleaned_text = clean_text(page_text)

        if not cleaned_text:
            continue

        # Step 4: Split cleaned text into chunks
        chunks = split_text(
            cleaned_text,
            chunk_size=500,
            overlap=100
        )

        # Step 5: Store chunks with metadata
        for chunk_number, chunk in enumerate(chunks, start=1):

            chunk_data = {
                "filename": file_path,
                "page_number": page["page_number"],
                "chunk_number": chunk_number,
                "text": chunk
            }

            all_chunks.append(chunk_data)

    return all_chunks


def main():

    print("===== IntelliAns Document Pipeline =====")

    file_path = "data/sample.pdf"

    chunks = process_document(file_path)

    print("Total chunks created:", len(chunks))

    for chunk in chunks:

        print("\n------------------------")
        print("Filename:", chunk["filename"])
        print("Page:", chunk["page_number"])
        print("Chunk:", chunk["chunk_number"])
        print("Text:", chunk["text"])
        print("Length:", len(chunk["text"]))


if __name__ == "__main__":
    main()
