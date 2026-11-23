import os
import tempfile
import sys
import textwrap
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st
import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer

# Ensure parent directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.process_document import process_document

# ------------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# ------------------------------------------------------------------------------

st.set_page_config(
    page_title="IntelliAns | AI Knowledge Assistant",
    page_icon="📚",
    layout="wide"
)

load_dotenv()

# ------------------------------------------------------------------------------
# 2. CUSTOM CSS (Matching Modern Obsidian / Dark Navy Theme & Top Padding Fix)
# ------------------------------------------------------------------------------

st.markdown("""
<style>
/* Main App Background */
.stApp {
    background-color: #0A1120 !important;
    color: #FFFFFF !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* Ensure generous top padding so header is never cut off */
.block-container {
    padding-top: 5.5rem !important;
    padding-bottom: 3.5rem !important;
    max-width: 920px !important;
}

/* Header bar styling */
header[data-testid="stHeader"] {
    background-color: transparent !important;
    height: 3.5rem !important;
}

/* Headings and labels */
h1, h2, h3, h4, p, span, label {
    color: #FFFFFF;
}

/* Feature Cards */
.feature-card {
    background: #111C30;
    border: 1px solid #1E2E4A;
    border-radius: 16px;
    padding: 22px 24px;
    min-height: 125px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    box-sizing: border-box;
}

.card-icon-pdf {
    width: 32px;
    height: 32px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
}

.card-icon-summary {
    width: 32px;
    height: 32px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
}

.card-title {
    color: #FFFFFF;
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 4px;
}

.card-subtitle {
    color: #94A3B8;
    font-size: 14px;
}

/* Streamlit Container Borders */
[data-testid="stVerticalBlockBorderWrapper"] > div {
    background-color: #111C30 !important;
    border: 1px solid #1E2E4A !important;
    border-radius: 16px !important;
    padding: 22px 24px !important;
}

/* Text Input Styling */
.stTextInput > div > div > input {
    background-color: #0E1729 !important;
    border: 1px solid #1E2E4A !important;
    color: #FFFFFF !important;
    border-radius: 10px !important;
    padding: 14px 16px !important;
    font-size: 15px !important;
}

.stTextInput > div > div > input:focus {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 1px #2563EB !important;
}

.stTextInput > div > div > input::placeholder {
    color: #64748B !important;
    font-size: 15px !important;
}

/* Primary Button (Generate Answer) */
.stButton > button[kind="primary"] {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 24px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    transition: background-color 0.2s ease;
}

.stButton > button[kind="primary"]:hover {
    background-color: #1D4ED8 !important;
}

/* Secondary Button (Process PDF / Action) */
.stButton > button[kind="secondary"] {
    background-color: #172B47 !important;
    color: #FFFFFF !important;
    border: 1px solid #2B4265 !important;
    border-radius: 10px !important;
    padding: 10px 18px !important;
    font-weight: 600 !important;
}

.stButton > button[kind="secondary"]:hover {
    background-color: #2563EB !important;
    border-color: #2563EB !important;
    color: #FFFFFF !important;
}

/* File Uploader */
[data-testid="stFileUploader"] {
    background-color: #0E1729 !important;
    border: 1px dashed #243854 !important;
    border-radius: 12px !important;
    padding: 12px !important;
}

[data-testid="stFileUploader"] section {
    background-color: transparent !important;
}

[data-testid="stFileUploader"] span, [data-testid="stFileUploader"] small {
    color: #94A3B8 !important;
}

/* Expanders */
[data-testid="stExpander"] {
    background-color: #111C30 !important;
    border: 1px solid #1E2E4A !important;
    border-radius: 12px !important;
    margin-bottom: 10px;
}

[data-testid="stExpander"] summary {
    color: #FFFFFF !important;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 3. LOAD EMBEDDING MODEL (CACHED)
# ------------------------------------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# ------------------------------------------------------------------------------
# 4. INITIALIZE SESSION STATE
# ------------------------------------------------------------------------------

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

# ------------------------------------------------------------------------------
# 5. TOP NAVBAR (Fully visible with proper margin & padding)
# ------------------------------------------------------------------------------

st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 0 20px 0; border-bottom: 1px solid rgba(255, 255, 255, 0.08); margin-bottom: 28px;">
    <!-- Logo and Brand -->
    <div style="display: flex; align-items: center; gap: 12px;">
        <div style="background: #2563EB; width: 38px; height: 38px; border-radius: 9px; display: flex; align-items: center; justify-content: center;">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-2.04Z"/>
                <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-4.44-2.04Z"/>
            </svg>
        </div>
        <span style="font-size: 21px; font-weight: 700; color: #FFFFFF; letter-spacing: -0.2px;">IntelliAns</span>
    </div>
    <!-- AI Assistant Pill Badge -->
    <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.35); color: #10B981; font-weight: 600; font-size: 13px; padding: 5px 14px; border-radius: 9999px; letter-spacing: 0.2px;">
        AI Assistant
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 6. GREETINGS & HERO SECTION
# ------------------------------------------------------------------------------

# Time-based client greetings
try:
    current_time = datetime.now(ZoneInfo("Asia/Kolkata"))
except Exception:
    current_time = datetime.now()

hour = current_time.hour

if hour < 12:
    greeting = "Good Morning"
    emoji = "☀️"
elif hour < 17:
    greeting = "Good Afternoon"
    emoji = "🌤️"
else:
    greeting = "Good Evening"
    emoji = "🌙"

st.markdown(f"""
<div style="margin-bottom: 24px;">
    <!-- Time-based greeting -->
    <div style="color: #60A5FA; font-size: 18px; font-weight: 700; margin-bottom: 4px; display: flex; align-items: center; gap: 8px;">
        <span>{emoji} {greeting}!</span>
    </div>
    <!-- Welcome message to client -->
    <div style="color: #FFFFFF; font-size: 22px; font-weight: 700; margin-bottom: 16px;">
        Welcome to IntelliAns 👋
    </div>
    <!-- Hero heading and subtitle -->
    <h1 style="font-size: 34px; font-weight: 800; color: #FFFFFF; margin: 0 0 8px 0; letter-spacing: -0.5px;">
        Your documents. Your answers.
    </h1>
    <p style="font-size: 16px; color: #94A3B8; margin: 0;">
        Upload, summarize and explore your documents with AI.
    </p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 7. TWO ACTION CARDS
# ------------------------------------------------------------------------------

col_card1, col_card2 = st.columns(2)

with col_card1:
    st.markdown("""
    <div class="feature-card">
        <div class="card-icon-pdf">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="#EF4444" stroke="#EF4444" stroke-width="1">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" fill="#EF4444" fill-opacity="0.2"/>
                <path d="M14 2v6h6M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6z" fill="none" stroke="#EF4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <text x="7" y="18" fill="#EF4444" font-size="6.5" font-weight="bold" font-family="sans-serif">PDF</text>
            </svg>
        </div>
        <div class="card-title">Upload PDF</div>
        <div class="card-subtitle">Add your documents</div>
    </div>
    """, unsafe_allow_html=True)

with col_card2:
    st.markdown("""
    <div class="feature-card">
        <div class="card-icon-summary">
            <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/>
                <circle cx="19" cy="5" r="1" fill="#F59E0B"/>
            </svg>
        </div>
        <div class="card-title">AI Summary</div>
        <div class="card-subtitle">Get key insights</div>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 8. UPLOAD PDF OPTION (PRESERVED & BEAUTIFULLY INTEGRATED)
# ------------------------------------------------------------------------------

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

with st.container(border=True):
    col_up_info, col_up_status = st.columns([3, 1])
    with col_up_info:
        st.markdown("""
            <div style="font-size: 15px; font-weight: 600; color: #FFFFFF; margin-bottom: 2px;">
                📄 Upload Your PDF Document
            </div>
            <div style="font-size: 13px; color: #94A3B8; margin-bottom: 10px;">
                Upload a PDF to extract text, build semantic vector embeddings, and explore with AI.
            </div>
        """, unsafe_allow_html=True)
    with col_up_status:
        if st.session_state.documents:
            st.markdown(
                '<div style="text-align: right; color: #10B981; font-size: 13px; font-weight: 600; padding-top: 4px;">✓ Document Ready</div>',
                unsafe_allow_html=True
            )

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        label_visibility="collapsed"
    )

    if st.session_state.documents:
        col_proc, col_sum, col_reset = st.columns([1, 1, 0.6])
        with col_proc:
            process_clicked = st.button("Process PDF", type="secondary", use_container_width=True)
        with col_sum:
            summary_clicked = st.button("✨ Generate AI Summary", use_container_width=True)
        with col_reset:
            if st.button("Clear / Reset", use_container_width=True):
                st.session_state.index = None
                st.session_state.documents = []
                st.session_state.answer = None
                st.session_state.sources = []
                st.session_state.summary = ""
                st.rerun()
    else:
        col_proc, col_sum = st.columns(2)
        with col_proc:
            process_clicked = st.button("Process PDF", type="secondary", use_container_width=True)
        with col_sum:
            summary_clicked = st.button("✨ Generate AI Summary", use_container_width=True)

# Process PDF logic
if process_clicked:
    if uploaded_file is None:
        st.warning("Please upload a PDF first.")
    else:
        try:
            with st.spinner("Processing your PDF..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_path = temp_file.name

                try:
                    documents = process_document(temp_path)
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                if not documents:
                    st.error("No readable text was found in this PDF.")
                    st.stop()

                for document in documents:
                    document["filename"] = uploaded_file.name

                model = load_embedding_model()
                texts = [document["text"] for document in documents]

                embeddings = model.encode(
                    texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                embeddings = np.asarray(embeddings, dtype=np.float32)

                index = faiss.IndexFlatIP(embeddings.shape[1])
                index.add(embeddings)

                st.session_state.index = index
                st.session_state.documents = documents
                st.session_state.answer = None
                st.session_state.sources = []
                st.session_state.summary = ""

            st.success(f"PDF processed successfully! Created {len(documents)} text chunks.")
        except Exception as error:
            st.error("Document processing failed.")
            st.exception(error)

# Generate AI Summary logic
if summary_clicked:
    if not st.session_state.documents:
        st.warning("Please upload and process a PDF first before generating a summary.")
    else:
        try:
            with st.spinner("Generating PDF summary using Gemini..."):
                pdf_text = "\n\n".join(document["text"] for document in st.session_state.documents)

                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    try:
                        api_key = st.secrets["GEMINI_API_KEY"]
                    except Exception:
                        api_key = None

                if not api_key:
                    st.error("Gemini API key not found. Please set GEMINI_API_KEY.")
                else:
                    client = genai.Client(
                        api_key=api_key,
                        http_options=types.HttpOptions(timeout=120000)
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

# ------------------------------------------------------------------------------
# 9. ASK INTELLIANS CARD (Exact format from screenshot)
# ------------------------------------------------------------------------------

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

with st.container(border=True):
    col_ask_title, col_ask_icon = st.columns([10, 1])
    with col_ask_title:
        st.markdown(
            '<div style="font-size: 18px; font-weight: 700; color: #FFFFFF; padding-top: 2px;">Ask IntelliAns</div>',
            unsafe_allow_html=True
        )
    with col_ask_icon:
        st.markdown(
            '''<div style="text-align: right;">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#60A5FA" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"></path>
                </svg>
            </div>''',
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    question = st.text_input(
        "Ask anything about your uploaded PDF...",
        placeholder="Ask anything about your uploaded PDF...",
        label_visibility="collapsed"
    )

    st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

    generate_clicked = st.button("Generate Answer →", type="primary", use_container_width=True)

# Generate answer logic
if generate_clicked:
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
                query_embedding = np.asarray(query_embedding, dtype=np.float32)

                index = st.session_state.index
                documents = st.session_state.documents
                top_k = min(3, index.ntotal)

                scores, indices = index.search(query_embedding, top_k)
                retrieved_chunks = []

                for score, idx in zip(scores[0], indices[0]):
                    if idx == -1:
                        continue
                    doc = documents[idx]
                    retrieved_chunks.append({
                        "text": doc["text"],
                        "filename": doc["filename"],
                        "page_number": doc["page_number"],
                        "chunk_number": doc["chunk_number"],
                        "similarity": float(score)
                    })

            if not retrieved_chunks:
                st.warning("No relevant information found.")
                st.stop()

            # Prepare context
            context_parts = []
            for number, chunk in enumerate(retrieved_chunks, start=1):
                context_parts.append(
                    f"Source {number}\nFilename: {chunk['filename']}\nPage: {chunk['page_number']}\nContent:\n{chunk['text']}\n"
                )
            context = "\n".join(context_parts)

            # Gemini answer generation
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                try:
                    api_key = st.secrets["GEMINI_API_KEY"]
                except Exception:
                    api_key = None

            if not api_key:
                st.error("Gemini API key not found. Configure GEMINI_API_KEY.")
                st.stop()

            client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=120000)
            )

            prompt = f"""
            You are IntelliAns, an intelligent document question-answering assistant.
            Answer the user's question using only the provided document context.

            Instructions:
            1. Give a clear and structured answer.
            2. Explain technical terms simply.
            3. Do not invent information.
            4. If the context is insufficient, say so.
            5. Mention the relevant PDF page numbers.
            6. Combine information from multiple sources when useful.

            DOCUMENT CONTEXT:
            {context}

            USER QUESTION:
            {question}

            ANSWER:
            """

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

# ------------------------------------------------------------------------------
# 10. DOCUMENT OVERVIEW (Exact format from screenshot)
# ------------------------------------------------------------------------------

st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

# Calculate statistics (0 at starting, changes dynamically according to PDFs added)
if st.session_state.documents:
    num_pdfs = len(set(d.get('filename', '') for d in st.session_state.documents))
    num_pages = len(set(f"{d.get('filename', '')}_{d.get('page_number', 0)}" for d in st.session_state.documents))
    num_chunks = len(st.session_state.documents)

    pdf_count = f"{num_pdfs:02d}"
    page_count = f"{num_pages:02d}"
    chunk_count = f"{num_chunks:02d}"
    
    filenames = list(dict.fromkeys(d.get("filename", "Active PDF") for d in st.session_state.documents))
    caption_text = f"Live document overview — {filenames[0]}"
else:
    pdf_count = "0"
    page_count = "0"
    chunk_count = "0"
    caption_text = "No document loaded — upload and process a PDF to view live metrics."

st.markdown(f"""
<div style="margin-bottom: 12px;">
    <span style="font-size: 13px; font-weight: 700; letter-spacing: 1.2px; color: #94A3B8; text-transform: uppercase;">
        DOCUMENT OVERVIEW
    </span>
</div>
<div style="display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; margin-bottom: 12px;">
    <div style="background: #FFFFFF; border-radius: 14px; padding: 22px 24px; box-shadow: 0 4px 14px rgba(0,0,0,0.06);">
        <div style="color: #2563EB; font-size: 32px; font-weight: 700; line-height: 1.1; margin-bottom: 4px;">{pdf_count}</div>
        <div style="color: #64748B; font-size: 14px; font-weight: 500;">PDFs</div>
    </div>
    <div style="background: #FFFFFF; border-radius: 14px; padding: 22px 24px; box-shadow: 0 4px 14px rgba(0,0,0,0.06);">
        <div style="color: #2563EB; font-size: 32px; font-weight: 700; line-height: 1.1; margin-bottom: 4px;">{page_count}</div>
        <div style="color: #64748B; font-size: 14px; font-weight: 500;">Pages</div>
    </div>
    <div style="background: #FFFFFF; border-radius: 14px; padding: 22px 24px; box-shadow: 0 4px 14px rgba(0,0,0,0.06);">
        <div style="color: #2563EB; font-size: 32px; font-weight: 700; line-height: 1.1; margin-bottom: 4px;">{chunk_count}</div>
        <div style="color: #64748B; font-size: 14px; font-weight: 500;">Chunks</div>
    </div>
</div>
<div style="font-size: 13px; color: #94A3B8; display: flex; align-items: center; gap: 8px; margin-top: 8px;">
    <span>{caption_text}</span>
    <span style="display: inline-flex; align-items: center; justify-content: center; width: 20px; height: 20px; border-radius: 50%; background: #000000; color: #FFFFFF; font-size: 11px;">↓</span>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 11. DISPLAY ANSWER & SOURCES (IF AVAILABLE)
# ------------------------------------------------------------------------------

if st.session_state.answer:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("### 💡 IntelliAns Answer")
        st.markdown(st.session_state.answer)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.subheader("Retrieved Sources")
    for number, source in enumerate(st.session_state.sources, start=1):
        with st.expander(f"Source {number} — Page {source['page_number']} (Score: {round(source['similarity'], 4)})"):
            st.write("**Filename:**", source["filename"])
            st.write("**Page Number:**", source["page_number"])
            st.write("**Chunk Number:**", source["chunk_number"])
            st.write("**Similarity Score:**", round(source["similarity"], 4))
            st.write("**Content:**")
            st.write(source["text"])

# ------------------------------------------------------------------------------
# 12. PDF GENERATION & DISPLAY SUMMARY (IF AVAILABLE)
# ------------------------------------------------------------------------------

def create_summary_pdf(title: str, text: str) -> bytes:
    """Generate a clean, professional, multi-page PDF document in pure Python."""
    page_width, page_height = 612, 792  # Standard Letter format
    margin_x, margin_top, margin_bottom = 54, 54, 54
    chars_per_line = 82

    # Clean text to safe Latin-1 / ASCII equivalents
    replacements = {
        '“': '"', '”': '"', '‘': "'", '’': "'",
        '—': '-', '–': '-', '…': '...', '•': '*',
        '\u00a0': ' ', '\t': '    '
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    safe_text = ''.join(c if (32 <= ord(c) <= 126 or c == '\n') else ' ' for c in text)

    lines = []
    for raw_line in safe_text.splitlines():
        line = raw_line.strip()
        if not line:
            lines.append(("", "spacing"))
            continue
        # Check if line is a heading
        if line.startswith("#") or (line.startswith("**") and line.endswith("**")):
            clean_head = line.strip("#* ").strip()
            lines.append((clean_head, "heading"))
        elif line.startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "- ", "* ")):
            wrapped = textwrap.wrap(line, width=chars_per_line)
            for idx, w in enumerate(wrapped):
                lines.append((w, "bullet" if idx == 0 else "bullet_cont"))
        else:
            wrapped = textwrap.wrap(line, width=chars_per_line)
            for w in wrapped:
                lines.append((w, "body"))

    # Paginate content
    pages_content = []
    current_page = []
    y = page_height - margin_top - 55

    for content, style in lines:
        if style == "heading":
            item_h = 24
        elif style == "spacing":
            item_h = 10
        else:
            item_h = 14

        if y - item_h < margin_bottom:
            pages_content.append(current_page)
            current_page = []
            y = page_height - margin_top - 55

        current_page.append((content, style, y))
        y -= item_h

    if current_page or not pages_content:
        pages_content.append(current_page)

    total_pages = len(pages_content)

    def escape_pdf(s: str) -> str:
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    num_pages = len(pages_content)
    page_ids = [5 + 2 * i for i in range(num_pages)]
    content_ids = [6 + 2 * i for i in range(num_pages)]

    catalog = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    kids_str = " ".join(f"{pid} 0 R" for pid in page_ids)
    pages_dict = f"2 0 obj\n<< /Type /Pages /Kids [{kids_str}] /Count {num_pages} >>\nendobj\n".encode("latin-1")
    font1 = b"3 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    font2 = b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>\nendobj\n"

    body_parts = [catalog, pages_dict, font1, font2]

    for i, page_items in enumerate(pages_content):
        pid = page_ids[i]
        cid = content_ids[i]

        page_obj = (
            f"{pid} 0 obj\n<< /Type /Page /Parent 2 0 R "
            f"/MediaBox [0 0 {page_width} {page_height}] "
            f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> "
            f"/Contents {cid} 0 R >>\nendobj\n"
        ).encode("latin-1")
        body_parts.append(page_obj)

        stream_cmds = [
            "q 0.145 0.388 0.922 rg",
            f"{margin_x} 742 504 2 re f Q",
            "BT /F2 14 Tf 0.05 0.1 0.25 rg",
            f"{margin_x} 750 Td ({escape_pdf(title)}) Tj ET",
            f"BT /F1 9 Tf 0.4 0.45 0.5 rg {page_width - margin_x - 65} 32 Td (Page {i+1} of {total_pages}) Tj ET",
            f"BT /F1 9 Tf 0.4 0.45 0.5 rg {margin_x} 32 Td (IntelliAns - AI Knowledge Assistant) Tj ET"
        ]

        for content, style, item_y in page_items:
            if not content:
                continue
            esc = escape_pdf(content)
            if style == "heading":
                stream_cmds.append(f"BT /F2 12 Tf 0.08 0.15 0.35 rg {margin_x} {item_y} Td ({esc}) Tj ET")
            elif style == "bullet":
                stream_cmds.append(f"BT /F2 10 Tf 0.15 0.2 0.3 rg {margin_x} {item_y} Td ({esc[:2]}) Tj ET")
                stream_cmds.append(f"BT /F1 10 Tf 0.15 0.2 0.3 rg {margin_x + 14} {item_y} Td ({escape_pdf(content[2:])}) Tj ET")
            elif style == "bullet_cont":
                stream_cmds.append(f"BT /F1 10 Tf 0.15 0.2 0.3 rg {margin_x + 14} {item_y} Td ({esc}) Tj ET")
            else:
                stream_cmds.append(f"BT /F1 10 Tf 0.15 0.2 0.3 rg {margin_x} {item_y} Td ({esc}) Tj ET")

        stream_data = "\n".join(stream_cmds).encode("latin-1")
        content_obj = (
            f"{cid} 0 obj\n<< /Length {len(stream_data)} >>\nstream\n".encode("latin-1")
            + stream_data
            + b"\nendstream\nendobj\n"
        )
        body_parts.append(content_obj)

    header = b"%PDF-1.4\n"
    offsets = []
    curr_offset = len(header)
    full_body = b""

    for part in body_parts:
        offsets.append(curr_offset)
        curr_offset += len(part)
        full_body += part

    xref_offset = curr_offset
    total_objs = len(offsets) + 1
    xref = f"xref\n0 {total_objs}\n0000000000 65535 f \n".encode("latin-1")
    for off in offsets:
        xref += f"{off:010d} 00000 n \n".encode("latin-1")

    trailer = f"trailer\n<< /Size {total_objs} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("latin-1")
    return header + full_body + xref + trailer


if st.session_state.summary:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("### 📄 AI Document Summary")
        st.markdown(st.session_state.summary)

        # Generate downloadable PDF
        pdf_summary_bytes = create_summary_pdf("IntelliAns Document Summary", st.session_state.summary)

        col_dl_pdf, col_dl_txt = st.columns([1, 1])
        with col_dl_pdf:
            st.download_button(
                "📥 Download Summary (PDF)",
                data=pdf_summary_bytes,
                file_name="IntelliAns_Summary.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True
            )
        with col_dl_txt:
            st.download_button(
                "📄 Download Summary (.txt)",
                data=st.session_state.summary,
                file_name="IntelliAns_Summary.txt",
                mime="text/plain",
                use_container_width=True
            )
