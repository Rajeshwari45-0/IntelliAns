
from pypdf import PdfReader


def extract_text_from_pdf(file_path):

    try:

        reader = PdfReader(file_path)

        extracted_pages = []

        for page_number, page in enumerate(reader.pages, start=1):

            text = page.extract_text() or ""

            page_data = {
                "page_number": page_number,
                "text": text
            }

            extracted_pages.append(page_data)

        return extracted_pages

    except FileNotFoundError:

        print("Error: PDF file not found.")
        return None

    except Exception as error:

        print("Error while reading PDF:", error)
        return None


def main():

    print("===== IntelliAns PDF Reader =====")

    file_path = "data/sample.pdf"

    pages = extract_text_from_pdf(file_path)

    if pages is not None:

        print("Total pages:", len(pages))

        for page in pages:

            print("\nPage:", page["page_number"])
            print(page["text"])


if __name__ == "__main__":

    main()
