
import os
import tempfile

import streamlit as st
import faiss
import numpy as np

from dotenv import load_dotenv
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.process_document import process_document


# -----------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------

st.set_page_config(
    page_title="IntelliAns | AI Knowledge Assistant",
    page_icon="📚",
    layout="wide"
)

load_dotenv()


# -----------------------------------------
# 2. LOAD EMBEDDING MODEL
# -----------------------------------------

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------------------
# 3. INITIALIZE SESSION STATE
# -----------------------------------------

if "index" not in st.session_state:
    st.session_state.index = None

if "documents" not in st.session_state:
    st.session_state.documents = []

if "answer" not in st.session_state:
    st.session_state.answer = None

if "sources" not in st.session_state:
    st.session_state.sources = []

if "summary" not in st.session_state:
    st.session_state.summary = ""


# -----------------------------------------
# 4. APPLICATION HEADER
# -----------------------------------------

st.title("📚 Welcome to IntelliAns")

st.subheader("AI-Powered Document Question Answering")

st.write(
    "Welcome to IntelliAns! Upload your PDF and ask questions. "
    "IntelliAns retrieves relevant information "
    "and generates answers using AI."
)

st.divider()


# -----------------------------------------
# 5. PDF UPLOAD
# -----------------------------------------

st.header("1. Upload Your Document")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)


# -----------------------------------------
# 6. PROCESS DOCUMENT
# -----------------------------------------

if st.button("Process PDF", type="primary"):

    if uploaded_file is None:

        st.warning("Please upload a PDF first.")

    else:

        try:

            with st.spinner("Processing your PDF..."):

                # Save uploaded PDF temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(uploaded_file.getvalue())
                    temp_path = temp_file.name

                try:

                    # Extract, clean and chunk the PDF
                    documents = process_document(temp_path)

                finally:

                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                if not documents:

                    st.error(
                        "No readable text was found in this PDF."
                    )

                    st.stop()

                # Preserve the original uploaded filename
                for document in documents:
                    document["filename"] = uploaded_file.name

                # Load embedding model
                model = load_embedding_model()

                # Convert document chunks into embeddings
                texts = [
                    document["text"]
                    for document in documents
                ]

                embeddings = model.encode(
                    texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )

                embeddings = np.asarray(
                    embeddings,
                    dtype=np.float32
                )

                # Create FAISS vector index
                index = faiss.IndexFlatIP(
                    embeddings.shape[1]
                )

                index.add(embeddings)

                # Save index and documents in session
                st.session_state.index = index
                st.session_state.documents = documents
                st.session_state.answer = None
                st.session_state.sources = []

            st.success("PDF processed successfully!")

            st.write("Total chunks created:", len(documents))
            st.write("Total vectors stored:", index.ntotal)

        except Exception as error:

            st.error("Document processing failed.")
            st.exception(error)


# -----------------------------------------
# 7. PDF SUMMARY GENERATOR
# -----------------------------------------

st.divider()

st.header("2. Generate PDF Summary")

if st.session_state.documents:

    if st.button("Generate PDF Summary"):

        try:
            with st.spinner("Generating PDF summary using Gemini..."):

                # Collect all extracted PDF text
                pdf_text = "\n\n".join(
                    document["text"]
                    for document in st.session_state.documents
                )

                # Get Gemini API key
                api_key = os.getenv("GEMINI_API_KEY")

                if not api_key:
                    try:
                        api_key = st.secrets["GEMINI_API_KEY"]
                    except Exception:
                        api_key = None

                if not api_key:
                    st.error("Gemini API key not found.")

                else:
                    client = genai.Client(
                        api_key=api_key,
                        http_options=types.HttpOptions(
                            timeout=120000
                        )
                    )

                    prompt = f"""
                    You are IntelliAns, a PDF summarization assistant.

                    Summarize the following PDF clearly.

                    Include:
                    1. Document Overview
                    2. Important Key Points
                    3. Main Findings
                    4. Conclusion

                    Use simple language. Do not invent information.

                    PDF CONTENT:
                    {pdf_text}
                    """

                    response = client.models.generate_content(
                        model="gemini-3.5-flash-lite",
                        contents=prompt
                    )

                    st.session_state.summary = response.text

        except Exception as error:
            st.error("Summary generation failed.")
            st.exception(error)

    # Display summary
    if st.session_state.summary:

        st.subheader("PDF Summary")

        st.markdown(st.session_state.summary)

        st.download_button(
            "Download Summary",
            data=st.session_state.summary,
            file_name="PDF_Summary.txt",
            mime="text/plain"
        )

else:
    st.info("Please upload and process a PDF first.")
# -----------------------------------------
# 7. QUESTION INPUT
# -----------------------------------------

st.divider()

st.header("2. Ask Your Question")

question = st.text_input(
    "Enter your question",
    placeholder="Example: What does MT mean?"
)


# -----------------------------------------
# 8. RETRIEVE RELEVANT CHUNKS
# -----------------------------------------

if st.button("Generate Answer", type="primary"):

    if st.session_state.index is None:

        st.warning("Please upload and process a PDF first.")

    elif not question.strip():

        st.warning("Please enter a question.")

    else:

        try:

            with st.spinner("Searching your document..."):

                model = load_embedding_model()

                query_embedding = model.encode(
                    [question],
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )

                query_embedding = np.asarray(
                    query_embedding,
                    dtype=np.float32
                )

                index = st.session_state.index
                documents = st.session_state.documents

                top_k = min(3, index.ntotal)

                scores, indices = index.search(
                    query_embedding,
                    top_k
                )

                retrieved_chunks = []

                for score, idx in zip(scores[0], indices[0]):

                    if idx == -1:
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

                st.warning("No relevant information found.")

                st.stop()

            # -----------------------------------------
            # 9. PREPARE DOCUMENT CONTEXT
            # -----------------------------------------

            context_parts = []

            for number, chunk in enumerate(
                retrieved_chunks,
                start=1
            ):

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

            # -----------------------------------------
            # 10. CONNECT TO GEMINI
            # -----------------------------------------

            api_key = os.getenv("GEMINI_API_KEY")

            if not api_key:

                try:
                    api_key = st.secrets["GEMINI_API_KEY"]
                except Exception:
                    api_key = None

            if not api_key:

                st.error(
                    "Gemini API key not found. "
                    "Configure GEMINI_API_KEY."
                )

                st.stop()

            client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(
                    timeout=120000
                )
            )

            prompt = f"""
You are IntelliAns, an intelligent document
question-answering assistant.

Answer the user's question using only the
provided document context.

Instructions:
1. Give a clear and structured answer.
2. Explain technical terms simply.
3. Do not invent information.
4. If the context is insufficient, say so.
5. Mention the relevant PDF page numbers.
6. Combine information from multiple sources
   when useful.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

            # -----------------------------------------
            # 11. GENERATE ANSWER
            # -----------------------------------------

            with st.spinner("Generating answer using Gemini..."):

                response = client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=prompt
                )

                answer = response.text

                if not answer:

                    st.error("Gemini returned an empty answer.")

                    st.stop()

            st.session_state.answer = answer
            st.session_state.sources = retrieved_chunks

        except Exception as error:

            st.error("Something went wrong.")

            st.exception(error)


# -----------------------------------------
# 12. DISPLAY ANSWER
# -----------------------------------------

if st.session_state.answer:

    st.divider()

    st.header("3. IntelliAns Answer")

    st.markdown(st.session_state.answer)

    # -----------------------------------------
    # 13. DISPLAY SOURCE REFERENCES
    # -----------------------------------------

    st.subheader("Retrieved Sources")

    for number, source in enumerate(
        st.session_state.sources,
        start=1
    ):

        with st.expander(f"Source {number}"):

            st.write("Filename:", source["filename"])

            st.write("Page:", source["page_number"])

            st.write("Chunk:", source["chunk_number"])

            st.write(
                "Similarity:",
                round(source["similarity"], 4)
            )

            st.write("Retrieved text:")

            st.write(source["text"])


# -----------------------------------------
# 14. FOOTER
# -----------------------------------------

st.divider()

st.caption(
    "IntelliAns | Intelligent AI-Powered Knowledge Assistant"
)
