import streamlit as st
import tempfile
import os
import numpy as np
import diskannpy
from sentence_transformers import SentenceTransformer
import PyPDF2

st.set_page_config(page_title="PDF Semantic Search Demo", layout="wide")
st.title("PDF Semantic Search with DiskANN")

# --- Upload PDF ---
st.header("1. Upload a PDF file")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    # --- Extract text from PDF ---
    st.header("2. Extracting text from PDF...")
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    all_text = "\n".join([page.extract_text() or "" for page in pdf_reader.pages])
    st.text_area("Extracted Text", all_text, height=200)

    # --- Split into sentences ---
    st.header("3. Generating embeddings...")
    sentences = [s.strip() for s in all_text.split(". ") if s.strip()]
    st.write(f"Found {len(sentences)} sentences.")

    # --- Generate embeddings ---
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sentences, convert_to_numpy=True, show_progress_bar=True)
    embeddings = embeddings.astype(np.float32)

    # --- Build DiskANN index in temp dir ---
    st.header("4. Building DiskANN index...")
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
    for i in range(num_points):
        index.insert(embeddings[i], i+1)  # 1-based IDs
    st.success("Index built!")

    # --- Search UI ---
    st.header("5. Semantic Search")
    query = st.text_input("Enter your search query:")
    if query:
        query_emb = model.encode([query], convert_to_numpy=True)[0].astype(np.float32)
        neighbors, distances = index.search(query_emb, k_neighbors=5, complexity=50)
        st.subheader("Top Results:")
        for idx, dist in zip(neighbors, distances):
            sent_idx = idx - 1  # 1-based to 0-based
            if 0 <= sent_idx < len(sentences):
                st.markdown(f"**{sentences[sent_idx]}**\n(Similarity: {1/(1+dist):.3f})")

    # Clean up temp file
    os.remove(pdf_path)
else:
    st.info("Please upload a PDF file to begin.")
