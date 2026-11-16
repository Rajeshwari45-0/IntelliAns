
import re


def clean_text(text):

    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)

    # Remove spaces from the beginning and end
    text = text.strip()

    return text


def main():

    sample_text = """
    IntelliAns     is an AI-powered knowledge assistant.


    It helps users     search documents.

    """

    cleaned_text = clean_text(sample_text)

    print("Original Text:")
    print(sample_text)

    print("\nCleaned Text:")
    print(cleaned_text)


if __name__ == "__main__":
    main()
