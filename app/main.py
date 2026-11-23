# IntelliAns - AI Knowledge Assistant

project_name = "IntelliAns"

def welcome_message():

    print("====================================")
    print("Welcome to", project_name)
    print("====================================")
    print("Building an AI-powered knowledge assistant")
    print("Using LLM, RAG and Machine Learning concepts")
    print("====================================\n")

# Display greetings to client at starting
welcome_message()

filename = input("Enter document filename: ")

file_type = input("Enter file type (PDF/DOCX/TXT): ")

pages = int(input("Enter number of pages: "))

document = {
    "filename": filename,
    "file_type": file_type,
    "pages": pages,
    "status": "Uploaded"
}

print("\nDocument Information")

print("Filename:", document["filename"])
print("File Type:", document["file_type"])
print("Pages:", document["pages"])
print("Status:", document["status"])