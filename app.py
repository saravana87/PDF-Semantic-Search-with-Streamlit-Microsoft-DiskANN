
import streamlit as st
import tempfile
import os
import numpy as np
import diskannpy
from sentence_transformers import SentenceTransformer
import PyPDF2

# --- Custom CSS for UI enhancement ---
st.markdown('''
    <style>
    .stApp { background-color: #f7fafd; }
    .main > div { background: #fff; border-radius: 12px; box-shadow: 0 2px 8px #e3e8ee; padding: 2rem; }
    .stTextInput > div > div > input { background: #f0f4f8; border-radius: 8px; }
    .stButton > button { background: #2563eb; color: white; border-radius: 8px; }
    .stTextArea textarea { background: #f0f4f8; border-radius: 8px; }
    .stMarkdown h2 { color: #2563eb; }
    .stMarkdown h3 { color: #0e7490; }
    .stMarkdown h4 { color: #0e7490; }
    .stAlert { border-radius: 8px; }
    </style>
''', unsafe_allow_html=True)


st.set_page_config(page_title="PDF Semantic Search Demo", layout="wide")
st.markdown("""
<h1 style='color:#0072c6; font-weight:900; letter-spacing:1px;'>📄 PDF Semantic Search <span style='color:#1a4e8a;'>with DiskANN</span></h1>
<p style='font-size:1.1em; color:#1a4e8a;'>Upload a PDF, index its content, and perform lightning-fast semantic search using state-of-the-art vector search (DiskANN) and sentence embeddings.</p>
""", unsafe_allow_html=True)



# --- Upload PDF ---
st.markdown("<h3 style='color:#0072c6;'>1. Upload a PDF file</h3>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    # --- Extract text from PDF ---
    st.markdown("<h3 style='color:#0072c6;'>2. Extracting text from PDF...</h3>", unsafe_allow_html=True)
    with st.spinner("Extracting text from PDF..."):
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        all_text = "\n".join([page.extract_text() or "" for page in pdf_reader.pages])
    st.text_area("Extracted Text", all_text, height=200)

    # --- Split into sentences ---
    st.markdown("<h3 style='color:#0072c6;'>3. Generating embeddings...</h3>", unsafe_allow_html=True)
    sentences = [s.strip() for s in all_text.split(". ") if s.strip()]
    st.write(f"<span style='color:#1a4e8a;'>Found <b>{len(sentences)}</b> sentences.</span>", unsafe_allow_html=True)

    # --- Generate embeddings ---
    with st.spinner("Generating embeddings..."):
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(sentences, convert_to_numpy=True, show_progress_bar=True)
        embeddings = embeddings.astype(np.float32)

    # --- Build DiskANN index in temp dir ---
    st.markdown("<h3 style='color:#0072c6;'>4. Building DiskANN index...</h3>", unsafe_allow_html=True)
    with st.spinner("Building DiskANN index..."):
        index_dir = tempfile.mkdtemp()
        dim = embeddings.shape[1]
        num_points = embeddings.shape[0]
        index = diskannpy.DynamicMemoryIndex(
            distance_metric="l2",
            vector_dtype=np.float32,
            dimensions=dim,
            max_vectors=num_points,
            complexity=50,
            graph_degree=32,
            alpha=1.2,
            num_threads=2
        )
        progress = st.progress(0, text="Inserting vectors into index...")
        for i in range(num_points):
            index.insert(embeddings[i], i+1)  # 1-based IDs
            if num_points > 0:
                progress.progress((i+1)/num_points, text=f"Inserted {i+1}/{num_points} vectors")
        progress.empty()
    st.success("Index built!")

    # --- Search UI ---
    st.markdown("<h3 style='color:#0072c6;'>5. Semantic Search</h3>", unsafe_allow_html=True)
    query = st.text_input("Enter your search query:")
    if query:
        with st.spinner("Searching..."):
            query_emb = model.encode([query], convert_to_numpy=True)[0].astype(np.float32)
            neighbors, distances = index.search(query_emb, k_neighbors=5, complexity=50)
        st.subheader(":blue[Top Results:]")
        for idx, dist in zip(neighbors, distances):
            sent_idx = idx - 1  # 1-based to 0-based
            if 0 <= sent_idx < len(sentences):
                st.markdown(f"""
                <div class='result-box'>
                <b style='color:#0072c6;'>{sentences[sent_idx]}</b><br>
                <span style='color:#1a4e8a;'>Similarity: <b>{1/(1+dist):.3f}</b></span>
                </div>
                """, unsafe_allow_html=True)

    # Clean up temp file
    os.remove(pdf_path)
else:
    st.info("Please upload a PDF file to begin.")
